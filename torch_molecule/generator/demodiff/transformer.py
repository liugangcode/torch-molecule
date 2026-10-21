import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .layers import Attention, MLP
from .conditions import TimestepEmbedder, ScoreEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def precompute_freqs_cis(
    dim: int, score: torch.Tensor, theta: float = 0.5, is_complex: bool = True
):
    """Compute rotary embeddings from similarity scores."""
    bs, n, _ = score.size()
    score = score.view(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=score.device).float() / dim))
    freqs = torch.outer(score, freqs)

    if is_complex:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis.view(bs, n, -1)
    else:
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        return torch.cat([cos_freqs, sin_freqs], dim=-1).view(bs, n, -1)


class Transformer(nn.Module):
    def __init__(
        self,
        max_n_nodes,
        hidden_size=384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.1,
        X_dim=118,
        E_dim=5,
        y_dim=3,
        pos_dim=50,
    ):
        super().__init__()

        self.E_total_dim = max_n_nodes * E_dim
        self.pos_total_dim = max_n_nodes * pos_dim

        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        assert hidden_size % 2 == 0
        atom_hidden = hidden_size // 2
        bond_hidden = hidden_size // 2
        pos_hidden = hidden_size // 2

        self.atom_embedder = nn.Linear(X_dim, atom_hidden, bias=False)
        self.bond_embedder = nn.Linear(E_dim * max_n_nodes, bond_hidden, bias=False)
        self.pos_embedder = nn.Linear(pos_dim * max_n_nodes, pos_hidden, bias=False)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.score_embedder = ScoreEmbedder(hidden_size, frequency_embedding_size=hidden_size)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        self.final_layer = FinalLayer(
            max_n_nodes=max_n_nodes,
            atom_type=X_dim,
            bond_type=E_dim,
            pos_dim=pos_dim,
            atom_hidden=atom_hidden,
            bond_hidden=bond_hidden,
            pos_hidden=pos_hidden,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.blocks:
            _constant_init(block.adaLN_modulation[-1], 0)

        _constant_init(self.final_layer.adaLN_modulation[-1], 0)

    def forward(self, X_in, E_in, pos_in, score, node_mask, target_n_nodes, t):
        bs, n, _ = X_in.size()
        left_E_dim = self.E_total_dim - E_in.size(-1)
        if left_E_dim > 0:
            E_in = F.pad(E_in, (0, left_E_dim), mode="constant", value=0)
        left_pos_dim = self.pos_total_dim - pos_in.size(-1)
        if left_pos_dim > 0:
            pos_in = F.pad(pos_in, (0, left_pos_dim), mode="constant", value=0)

        x_atom = self.atom_embedder(X_in)
        x_bond = self.bond_embedder(E_in)
        x_pos = self.pos_embedder(pos_in)
        x = torch.cat([x_atom, x_bond + x_pos], dim=-1)
        assert x.shape == (bs, n, self.hidden_size)

        score_emb = self.score_embedder(score)
        t_emb = self.t_embedder(t).unsqueeze(1)
        c = t_emb + score_emb

        freqs_cis = precompute_freqs_cis(self.head_dim, score)

        for i, block in enumerate(self.blocks):
            x = block(x, c, freqs_cis, node_mask["all"])

        X, E, pos = self.final_layer(x, X_in, E_in, pos_in, c, node_mask["target"], target_n_nodes)
        out = utils.PlaceHolder(X=X, E=E, y=None, pos=pos).mask(node_mask["target"])
        return out.X, out.E, out.pos


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, **block_kwargs
        )

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.0,
            act_layer=approx_gelu,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, freqs_cis, node_mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis, node_mask=node_mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    def __init__(
        self, max_n_nodes, atom_hidden, bond_hidden, pos_hidden, atom_type, bond_type, pos_dim
    ):
        super().__init__()

        self.atom_type = atom_type
        self.bond_type = bond_type
        self.pos_dim = pos_dim

        self.atom_hidden = atom_hidden
        self.bond_hidden = bond_hidden
        self.pos_hidden = pos_hidden
        hidden_size = atom_hidden + bond_hidden

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear_atom = nn.Linear(atom_hidden, atom_type, bias=False)
        self.linear_bond = nn.Linear(bond_hidden, max_n_nodes * bond_type, bias=False)
        self.linear_pos = nn.Linear(pos_hidden, max_n_nodes * pos_dim, bias=False)

    def forward(self, x, X_in, E_in, pos_in, c, node_mask, N):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = x[:, :N, :]
        B, N, D = x.size()

        atom_hidden = x[:, :, : self.atom_hidden]
        bond_hidden = x[:, :, self.atom_hidden :]
        pos_hidden = bond_hidden

        atom_out = self.linear_atom(atom_hidden)
        bond_out = self.linear_bond(bond_hidden)[:, :, : N * self.bond_type].reshape(
            B, N, N, self.bond_type
        )
        pos_out = self.linear_pos(pos_hidden)[:, :, : N * self.pos_dim].reshape(
            B, N, N, self.pos_dim
        )

        # Trim padding before reshaping residuals to the current batch width.
        bond_out = E_in[:, :N, : N * self.bond_type].reshape(B, N, N, self.bond_type) + bond_out

        pos_out = pos_in[:, :N, : N * self.pos_dim].reshape(B, N, N, self.pos_dim) + pos_out

        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1).type_as(edge_mask)
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        pos_out.masked_fill_(edge_mask[:, :, :, None], 0)
        pos_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))
        return atom_out, bond_out, pos_out
