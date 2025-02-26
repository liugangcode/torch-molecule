import torch
import utils
from diffusion import diffusion_utils
    
class NoiseScheduleDiscrete(torch.nn.Module):
    def __init__(self, timesteps):
        super(NoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
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
        ### new
        self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]
    
class MarginalTransition:
    def __init__(self, x_marginals, e_marginals, xe_conditions, ex_conditions, y_classes, n_nodes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
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

        return utils.PlaceHolder(X=q, E=None, y=None)
    
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

        return utils.PlaceHolder(X=q, E=None, y=None)