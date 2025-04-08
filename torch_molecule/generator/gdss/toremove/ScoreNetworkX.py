import torch
import torch.nn.functional as F

from .layers import DenseGCNConv, MLP, AttentionLayer
from .utils import mask_x, pow_tensor

class ScoreNetworkX(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid):
        super(ScoreNetworkX, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):

        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x


# class ScoreNetworkX_GMH(torch.nn.Module):
#     def __init__(self, max_feat_num, depth, nhid, num_linears,
#                  c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
#         super().__init__()

#         self.depth = depth
#         self.c_init = c_init

#         self.layers = torch.nn.ModuleList()
#         for _ in range(self.depth):
#             if _ == 0:
#                 self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
#                                                   c_hid, num_heads, conv))
#             elif _ == self.depth - 1:
#                 self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
#                                                   c_final, num_heads, conv))
#             else:
#                 self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
#                                                   c_hid, num_heads, conv))

#         fdim = max_feat_num + depth * nhid
#         self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
#                          use_bn=False, activate_func=F.elu)

#         self.activation = torch.tanh

#     def forward(self, x, adj, flags):
#         adjc = pow_tensor(adj, self.c_init)

#         x_list = [x]
#         for _ in range(self.depth):
#             x, adjc = self.layers[_](x, adjc, flags)
#             x = self.activation(x)
#             x_list.append(x)

#         xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
#         out_shape = (adj.shape[0], adj.shape[1], -1)
#         x = self.final(xs).view(*out_shape)
#         x = mask_x(x, flags)

#         return x