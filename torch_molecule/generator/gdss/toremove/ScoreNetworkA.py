import torch
from torch.nn import Parameter
import torch.nn.functional as F

from .utils import mask_adjs, pow_tensor, mask_x
from .layers import DenseGCNConv, MLP, AttentionLayer

# def node_feature_to_matrix(x):
#     """
#     :param x:  BS x N x F
#     :return:
#     x_pair: BS x N x N x 2F
#     """
#     x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
#     x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F
#     return x_pair

# class BaselineNetworkLayer(torch.nn.Module):

#     def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

#         super(BaselineNetworkLayer, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         for _ in range(input_dim):
#             self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
#         self.hidden_dim = max(input_dim, output_dim)
#         self.mlp_in_dim = input_dim + 2*conv_output_dim
#         self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
#                             use_bn=False, activate_func=F.elu)
#         self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
#                                     use_bn=False, activate_func=F.elu)
        
#     def forward(self, x, adj, flags):
    
#         x_list = []
#         for _ in range(len(self.convs)):
#             _x = self.convs[_](x, adj[:,_,:,:])
#             x_list.append(_x)
#         x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)) , flags)
#         x_out = torch.tanh(x_out)

#         x_matrix = node_feature_to_matrix(x_out)
#         mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
#         shape = mlp_in.shape
#         mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
#         _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
#         _adj = _adj + _adj.transpose(-1,-2)
#         adj_out = mask_adjs(_adj, flags)

#         return x_out, adj_out


# class BaselineNetwork(torch.nn.Module):

#     def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
#                     c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

#         super(BaselineNetwork, self).__init__()

#         self.nfeat = max_feat_num
#         self.max_node_num = max_node_num
#         self.nhid  = nhid
#         self.num_layers = num_layers
#         self.num_linears = num_linears
#         self.c_init = c_init
#         self.c_hid = c_hid
#         self.c_final = c_final

#         self.layers = torch.nn.ModuleList()
#         for _ in range(self.num_layers):
#             if _==0:
#                 self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

#             elif _==self.num_layers-1:
#                 self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

#             else:
#                 self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

#         self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
#         self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
#                             use_bn=False, activate_func=F.elu)
#         self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
#         self.mask.unsqueeze_(0)   

#     def forward(self, x, adj, flags=None):

#         adjc = pow_tensor(adj, self.c_init)

#         adj_list = [adjc]
#         for _ in range(self.num_layers):

#             x, adjc = self.layers[_](x, adjc, flags)
#             adj_list.append(adjc)
        
#         adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
#         out_shape = adjs.shape[:-1] # B x N x N
#         score = self.final(adjs).view(*out_shape)

#         self.mask = self.mask.to(score.device)
#         score = score * self.mask

#         score = mask_adjs(score, flags)

#         return score


# class ScoreNetworkA(BaselineNetwork):
    # def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
    #                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
    #     super(ScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
    #                                         c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')
class ScoreNetworkA(torch.nn.Module):
    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super(ScoreNetworkA, self).__init__()
        
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.nfeat = max_feat_num
        self.nhid = nhid
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.max_node_num = max_node_num

        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
            elif i==self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)  

    def forward(self, x, adj, flags):

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for i in range(self.num_layers):
            x, adjc = self.layers[i](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)
        
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score