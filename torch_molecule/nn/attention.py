from torch.jit import Final
import torch.nn.functional as F
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn

class AttentionWithNodeMask(nn.Module):
    fast_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_head=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_head == 0, 'dim should be divisible by num_head'
        self.num_head = num_head
        self.head_dim = dim // num_head

        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME
        assert self.fast_attn, "scaled_dot_product_attention Not implemented"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, node_mask):
        B, N, D = x.shape
        
        # B, head, N, head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # B, head, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)
        
        attn_mask = (node_mask[:, None, :, None] & node_mask[:, None, None, :]).expand(-1, self.num_head, N, N)
        attn_mask = attn_mask.clone()
        attn_mask[attn_mask.sum(-1) == 0] = True

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
