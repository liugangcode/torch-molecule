import torch
import torch.nn as nn
import torch.nn.functional as F

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
        input_dim_X=118,
        input_dim_E=5,
        input_dim_y=None,
        task_type=[], # a list of 'regression' or 'classification'
    ):
        super().__init__()
        self.input_dim_y = input_dim_y
        self.x_embedder = nn.Linear(input_dim_X + max_node * input_dim_E, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if input_dim_y > 0 and len(task_type) > 0:
            self.y_embedder_list = torch.nn.ModuleList()
            for i in range(input_dim_y):
                if task_type[i] == 'regression':
                    self.y_embedder_list.append(ClusterContinuousEmbedder(1, hidden_size, drop_condition))
                else:
                    self.y_embedder_list.append(CategoricalEmbedder(2, hidden_size, drop_condition))
        else:
            self.y_embedder_list = None

        self.blocks = nn.ModuleList(
            [
                AttentionBlock(hidden_size, num_head, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layer)
            ]
        )

        self.final_layer = FinalLayer(
            max_node=max_node,
            hidden_size=hidden_size,
            atom_type=input_dim_X,
            bond_type=input_dim_E,
            mlp_ratio=mlp_ratio,
            num_head=num_head,
        )

        self.initialize_parameters()

    def initialize_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

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

    def forward(self, noisy_data, unconditioned):
        X, E = noisy_data['X_t'].float(), noisy_data['E_t'].float()
        X_in, E_in = X, E
        node_mask, t =  noisy_data['node_mask'], noisy_data['t']
        y = noisy_data['y_t'] if self.y_embedder_list is not None else None
        assert (y is None) == (self.y_embedder_list is None), "y and y_embedder_list must both be None or both be not None"

        if y is not None:
            force_drop_id = torch.zeros_like(y.sum(-1))
            force_drop_id[torch.isnan(y.sum(-1))] = 1
            if unconditioned:
                force_drop_id = torch.ones_like(y[:, 0])
        
        bs, n, _ = X.size()
        h = torch.cat([X, E.reshape(bs, n, -1)], dim=-1)
        h = self.x_embedder(h)
        c = self.t_embedder(t)
        if self.y_embedder_list is not None:
            for i in range(self.input_dim_y):
                c = c + self.y_embedder_list[i](y[:, i:i+1], self.training, force_drop_id)
        
        for i, block in enumerate(self.blocks):
            h = block(h, c, node_mask)

        X, E, y = self.final_layer(h, X_in, E_in, c, t, node_mask)
        return PlaceHolder(X=X, E=E, y=y).mask(node_mask)
    
    def compute_loss(self, noisy_data, true_X, true_E, lw_X, lw_E, unconditioned=False):
        pred = self.forward(noisy_data, unconditioned=unconditioned)
        
        # Reshape predictions and targets
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(pred.X, (-1, pred.X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        
        # Calculate node and edge losses using cross entropy
        loss_X = F.cross_entropy(flat_pred_X, torch.argmax(flat_true_X, dim=-1)) if true_X.numel() > 0 else 0.0
        loss_E = F.cross_entropy(flat_pred_E, torch.argmax(flat_true_E, dim=-1)) if true_E.numel() > 0 else 0.0
        loss = lw_X * loss_X + lw_E * loss_E
        return loss, loss_X, loss_E

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
            use_bn=False,
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
        self.XEdecoder = MLP(in_features=hidden_size, out_features=final_size, drop=0., use_bn=False)

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, c, t, node_mask):
        x_all = self.XEdecoder(x)
        bs, n, _ = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        
        atom_out = x_all[:, :, :self.atom_type]
        atom_out = x_in + atom_out

        bond_out = x_all[:, :, self.atom_type:].reshape(bs, n, n, self.bond_type)
        bond_out = e_in + bond_out

        ##### standardize adj_out
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(n, dtype=torch.bool)
            .unsqueeze(0)
            .expand(bs, -1, -1)
            .type_as(edge_mask)
        )
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)
        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))
        return atom_out, bond_out, None