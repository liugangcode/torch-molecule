import numpy as np
import torch
from torch.nn import functional as F

from .utils import PlaceHolder


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def linear_beta_schedule_discrete(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule from beta_start to beta_end over timesteps."""
    return np.linspace(beta_start, beta_end, timesteps + 1)


def custom_beta_schedule_discrete(timesteps, average_num_nodes=30, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def sample_discrete_features(probX, probE, prob_pos, node_mask, exclude_null_pos=False):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape

    probX[~node_mask] = 1 / probX.shape[-1]
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    probX = probX + 1e-12
    probX = probX / probX.sum(dim=-1, keepdim=True)
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=probE.device).unsqueeze(0).expand(bs, -1, -1)
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    probE = probE + 1e-12
    probE = probE / probE.sum(dim=-1, keepdim=True)
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    prob_pos[inverse_edge_mask] = 1 / prob_pos.shape[-1]
    prob_pos[diag_mask.bool()] = 1 / prob_pos.shape[-1]
    prob_pos = prob_pos.reshape(bs * n * n, -1)  # (bs * n * n, dpos_out)
    if exclude_null_pos:
        prob_pos[:, 0] = 0
    prob_pos = prob_pos + 1e-12
    prob_pos = prob_pos / prob_pos.sum(dim=-1, keepdim=True)
    pos_t = prob_pos.multinomial(1).reshape(bs, n, n)  # (bs, n, n)

    return PlaceHolder(X=X_t, E=E_t, pos=pos_t)


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


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).type_as(x_limit)

    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).type_as(x_limit)

    pos_limit = limit_dist.pos[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_pos = pos_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_pos = F.one_hot(U_pos.long(), num_classes=pos_limit.shape[-1]).type_as(x_limit)

    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, pos=U_pos).mask(node_mask)
