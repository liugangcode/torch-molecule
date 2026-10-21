import torch
from . import utils
from . import diffusion as diffusion_utils


class NoiseScheduleDiscrete(torch.nn.Module):
    def __init__(self, noise_schedule, timesteps):
        super(NoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "custom":
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        elif noise_schedule == "linear":
            betas = diffusion_utils.linear_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer("betas", torch.from_numpy(betas).float())

        self.register_buffer("alphas", 1 - torch.clamp(self.betas, min=0, max=1))

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.register_buffer("alphas_bar", torch.exp(log_alpha_bar))

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
    def __init__(
        self, x_marginals, e_marginals, xe_conditions, ex_conditions, pos_marginals, n_nodes
    ):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.x_marginals = x_marginals  # Dx
        self.e_marginals = e_marginals  # Dx, De
        self.xe_conditions = xe_conditions
        self.pos_marginals = pos_marginals
        self.pos_dim = pos_marginals.size(0)

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)  # 1, Dx, Dx
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)  # 1, De, De
        self.u_xe = xe_conditions.unsqueeze(0)  # 1, Dx, De
        self.u_ex = ex_conditions.unsqueeze(0)  # 1, De, Dx
        self.u_pos = (
            pos_marginals.unsqueeze(0).expand(self.pos_dim, -1).unsqueeze(0)
        )  # 1, Dpos, Dpos
        self.n_nodes = n_nodes

    def get_union_transition(self, u_x, u_e, u_xe, u_ex, n_nodes):
        u_e = u_e.repeat(1, n_nodes, n_nodes)  # (1, n*de, n*de)
        u_xe = u_xe.repeat(1, 1, n_nodes)  # (1, dx, n*de)
        u_ex = u_ex.repeat(1, n_nodes, 1)  # (1, n*de, dx)
        u0 = torch.cat([u_x, u_xe], dim=2)  # (1, dx, dx + n*de)
        u1 = torch.cat([u_ex, u_e], dim=2)  # (1, n*de, dx + n*de)
        u = torch.cat([u0, u1], dim=1)  # (1, dx + n*de, dx + n*de)
        return u

    def index_edge_margin(self, X, q_e, n_bond=5):
        bs, n, n_atom = X.shape
        node_indices = X.argmax(-1)  # (bs, n)
        ind = node_indices[:, :, None].expand(bs, n, n_bond)
        q_e = torch.gather(q_e, 1, ind)
        return q_e

    def get_Qt(self, beta_t, device, n_nodes=None):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)
        returns: q (bs, d_XE, d_XE)
        """
        num_nodes = n_nodes if n_nodes else self.n_nodes
        bs = beta_t.size(0)
        u_XE = self.get_union_transition(self.u_x, self.u_e, self.u_xe, self.u_ex, num_nodes)
        d_XE = u_XE.size(-1)
        u_XE = u_XE.to(device).expand(bs, d_XE, d_XE)
        beta_t = beta_t.to(device)
        beta_t = beta_t.view(bs, 1, 1)
        q_XE = beta_t * u_XE + (1 - beta_t) * torch.eye(d_XE, device=device).unsqueeze(0)

        self.u_pos = self.u_pos.to(device)
        pos = beta_t * self.u_pos + (1 - beta_t) * torch.eye(self.pos_dim, device=device).unsqueeze(
            0
        )

        return utils.PlaceHolder(X=q_XE, E=None, y=None, pos=pos)

    def get_Qt_bar(self, alpha_bar_t, device, n_nodes=None):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K
        alpha_bar_t: (bs, 1) roduct of the (1 - beta_t) for each time step from 0 to t.
        returns: q (bs, d_XE, d_XE)
        """
        num_nodes = n_nodes if n_nodes else self.n_nodes
        u_XE = self.get_union_transition(self.u_x, self.u_e, self.u_xe, self.u_ex, num_nodes)
        d_XE = u_XE.size(-1)
        u_XE = u_XE.to(device)
        bs = alpha_bar_t.size(0)
        alpha_bar_t = alpha_bar_t.to(device)
        alpha_bar_t = alpha_bar_t.view(bs, 1, 1)
        q_XE = alpha_bar_t * torch.eye(d_XE, device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_XE

        self.u_pos = self.u_pos.to(device)
        # The original position schedule uses the opposite coefficients to node/edge noise.
        pos = alpha_bar_t * self.u_pos + (1 - alpha_bar_t) * torch.eye(
            self.pos_dim, device=device
        ).unsqueeze(0)

        return utils.PlaceHolder(X=q_XE, E=None, y=None, pos=pos)
