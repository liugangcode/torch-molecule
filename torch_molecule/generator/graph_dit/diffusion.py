import torch
import numpy as np
from torch.nn import functional as F
from .utils import PlaceHolder

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

class NoiseScheduleDiscrete(torch.nn.Module):
    def __init__(self, timesteps):
        super(NoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule_discrete(timesteps)
        self.register_buffer('betas', torch.from_numpy(betas).float())

        # 0.9999
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=1)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]
    
class MarginalTransition:
    def __init__(self, x_marginals, e_marginals, xe_conditions, ex_conditions, n_nodes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.x_marginals = x_marginals # Dx
        self.e_marginals = e_marginals # Dx, De
        self.xe_conditions = xe_conditions

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0) # 1, Dx, Dx
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0) # 1, De, De
        self.u_xe = xe_conditions.unsqueeze(0) # 1, Dx, De
        self.u_ex = ex_conditions.unsqueeze(0) # 1, De, Dx
        self.u = self.get_union_transition(self.u_x, self.u_e, self.u_xe, self.u_ex, n_nodes) # 1, Dx + n*De, Dx + n*De

    def get_union_transition(self, u_x, u_e, u_xe, u_ex, n_nodes):
        u_e = u_e.repeat(1, n_nodes, n_nodes) # (1, n*de, n*de)
        u_xe = u_xe.repeat(1, 1, n_nodes) # (1, dx, n*de)
        u_ex = u_ex.repeat(1, n_nodes, 1) # (1, n*de, dx)
        u0 = torch.cat([u_x, u_xe], dim=2) # (1, dx, dx + n*de)
        u1 = torch.cat([u_ex, u_e], dim=2) # (1, n*de, dx + n*de)
        u = torch.cat([u0, u1], dim=1) # (1, dx + n*de, dx + n*de)
        return u

    def index_edge_margin(self, X, q_e, n_bond=5):
        # q_e: (bs, dx, de) --> (bs, n, de)
        bs, n, n_atom = X.shape
        node_indices = X.argmax(-1)  # (bs, n)
        ind = node_indices[ :, :, None].expand(bs, n, n_bond)
        q_e = torch.gather(q_e, 1, ind)
        return q_e
    
    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)
        returns: q (bs, d0, d0)
        """
        bs = beta_t.size(0)
        d0 = self.u.size(-1)
        self.u = self.u.to(device)
        u = self.u.expand(bs, d0, d0)

        beta_t = beta_t.to(device)
        beta_t = beta_t.view(bs, 1, 1)
        q = beta_t * u + (1 - beta_t) * torch.eye(d0, device=device).unsqueeze(0)

        return PlaceHolder(X=q, E=None, y=None)
    
    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K
        alpha_bar_t: (bs, 1) roduct of the (1 - beta_t) for each time step from 0 to t.
        returns: q (bs, d0, d0)
        """
        bs = alpha_bar_t.size(0)
        d0 = self.u.size(-1)
        alpha_bar_t = alpha_bar_t.to(device)
        alpha_bar_t = alpha_bar_t.view(bs, 1, 1)
        self.u = self.u.to(device)
        q = alpha_bar_t * torch.eye(d0, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u

        return PlaceHolder(X=q, E=None, y=None)

def sample_discrete_features(probX, probE, node_mask):
    """ Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape

    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    probX = probX + 1e-12
    probX = probX / probX.sum(dim=-1, keepdim=True)
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)
    probE = probE + 1e-12
    probE = probE / probE.sum(dim=-1, keepdim=True)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t, y=None)

def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).type_as(x_limit)

    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).type_as(x_limit)

    U_X = U_X.to(node_mask.device)
    U_E = U_E.to(node_mask.device)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, y=None).mask(node_mask)

def reverse_diffusion(predX_0, X_t, Qt, Qsb, Qtb):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    """
    Qt_T = Qt.transpose(-1, -2)  # bs, N, dt
    assert Qt.dim() == 3
    left_term = X_t @ Qt_T  # bs, N, d_t-1
    right_term = predX_0 @ Qsb
    numerator = left_term * right_term  # bs, N, d_t-1

    denominator = Qtb @ X_t.transpose(-1, -2)  # bs, d0, N
    denominator = denominator.transpose(-1, -2)  # bs, N, d0
    return numerator / denominator.clamp_min(1e-5)
