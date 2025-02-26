import torch
import torch.nn as nn

from .utils import PlaceHolder
from ...nn import MLP, AttentionWithNodeMask
from ...nn.embedder import TimestepEmbedder, CategoricalEmbedder, ClusterContinuousEmbedder

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Transformer(nn.Module):
    def __init__(
        self,
        max_node,
        hidden_size=384,
        num_layer=12,
        num_head=16,
        mlp_ratio=4.0,
        dropout=0.,
        drop_condition=0.1,
        Xdim=118,
        Edim=5,
        ydim=3,
        task_type=[], # 'regression' or 'classification'
    ):
        super().__init__()
        self.ydim = ydim
        self.x_embedder = nn.Linear(Xdim + max_node * Edim, hidden_size, bias=False)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_list = torch.nn.ModuleList()

        for i in range(ydim):
            if task_type[i] == 'regression':
                self.y_embedder_list.append(ClusterContinuousEmbedder(1, hidden_size, drop_condition))
            else:
                self.y_embedder_list.append(CategoricalEmbedder(2, hidden_size, drop_condition))

        self.blocks = nn.ModuleList(
            [
                AttentionBlock(hidden_size, num_head, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layer)
            ]
        )

        self.final_layer = FinalLayer(
            max_node=max_node,
            hidden_size=hidden_size,
            atom_type=Xdim,
            bond_type=Edim,
            mlp_ratio=mlp_ratio,
            num_head=num_head,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
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

        for block in self.blocks :
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.final_layer.adaLN_modulation[0], 0)

    def forward(self, x, e, node_mask, y, t, unconditioned):
        force_drop_id = torch.zeros_like(y.sum(-1))
        force_drop_id[torch.isnan(y.sum(-1))] = 1
        if unconditioned:
            force_drop_id = torch.ones_like(y[:, 0])
        
        x_in, e_in, y_in = x, e, y
        bs, n, _ = x.size()
        x = torch.cat([x, e.reshape(bs, n, -1)], dim=-1)
        x = self.x_embedder(x)
        c = self.t_embedder(t)
        for i in range(self.ydim):
            c = c + self.y_embedder_list[i](y[:, i:i+1], self.training, force_drop_id, t)
        
        for i, block in enumerate(self.blocks):
            x = block(x, c, node_mask)

        X, E, y = self.final_layer(x, x_in, e_in, c, t, node_mask)
        return PlaceHolder(X=X, E=E, y=y).mask(node_mask)

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_head, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = AttentionWithNodeMask(
            hidden_size, num_head=num_head, qkv_bias=True, qk_norm=True, proj_drop=dropout
        )
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=self.dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c, node_mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * modulate(self.norm1(self.attn(x, node_mask=node_mask)), shift_msa, scale_msa)
        x = x + gate_mlp.unsqueeze(1) * modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x


class FinalLayer(nn.Module):
    def __init__(self, max_node, hidden_size, atom_type, bond_type, mlp_ratio, num_head=None):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        final_size = atom_type + max_node * bond_type
        self.XEdecoder = MLP(in_features=hidden_size, 
                            out_features=final_size, drop=0)

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, c, t, node_mask):
        x_all = self.XEdecoder(x)
        B, N, D = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        
        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out

        bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)
        bond_out = e_in + bond_out

        ##### standardize adj_out
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))
        return atom_out, bond_out, None