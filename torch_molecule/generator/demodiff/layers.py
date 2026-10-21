from torch.jit import Final
import torch.nn.functional as F

import torch
import torch.nn as nn


def apply_rotary_emb(q, k, freqs_cis):
    """Apply rotary positional embeddings to query and key tensors."""
    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    q_ = torch.view_as_complex(q_reshaped)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
    k_ = torch.view_as_complex(k_reshaped)
    freqs_cis_expanded = freqs_cis[:, None, :, :]
    q_rot = q_ * freqs_cis_expanded
    k_rot = k_ * freqs_cis_expanded
    q_real = torch.view_as_real(q_rot)
    q_out = q_real.flatten(3)
    k_real = torch.view_as_real(k_rot)
    k_out = k_real.flatten(3)
    q_final = q_out.type_as(q)
    k_final = k_out.type_as(k)

    return q_final, k_final


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0,
        proj_drop=0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.fast_attn, "scaled_dot_product_attention Not implemented"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def dot_product_attention(self, q, k, v):
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_sfmx = attn.softmax(dim=-1)
        attn_sfmx = self.attn_drop(attn_sfmx)
        x = attn_sfmx @ v
        return x, attn

    def forward(self, x, freqs_cis, node_mask):
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, head, N, head_dim
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k = self.q_norm(q), self.k_norm(k)
        attn_mask = (
            (node_mask[:, None, :, None] & node_mask[:, None, None, :])
            .expand(-1, self.num_heads, N, N)
            .clone()
        )
        attn_mask[attn_mask.sum(-1) == 0] = True

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
