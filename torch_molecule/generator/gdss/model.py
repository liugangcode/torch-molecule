import torch
import torch.nn.functional as F

from .sde import get_score_fn
from .utils import mask_adjs, pow_tensor, mask_x, gen_noise
from .layers import DenseGCNConv, MLP, AttentionLayer

# class GDSSModel(torch.nn.Module):
#     def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
#                     c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
#         super(GDSSModel, self).__init__()

#         self.score_network_a = ScoreNetworkA(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
#                                             c_init, c_hid, c_final, adim, num_heads, conv)
#         self.score_network_x = ScoreNetworkX(max_feat_num, num_layers, nhid)

class GDSSModel(torch.nn.Module):
    def __init__(self, input_dim_X, max_node, hidden_size, num_layer, input_dim_adj, hidden_size_adj, attention_dim, num_head=4, conv='GCN'):
        super(GDSSModel, self).__init__()
        num_layer_in_mlp = 3

        self.score_network_a = ScoreNetworkA(
            max_feat_num=input_dim_X,
            max_node_num=max_node,
            nhid=hidden_size,
            num_layers=num_layer,
            num_linears=num_layer_in_mlp,
            c_init=input_dim_adj,
            c_hid=hidden_size_adj,
            c_final=input_dim_adj,
            adim=attention_dim,
            num_heads=num_head,
            conv=conv
        )
        self.score_network_x = ScoreNetworkX(
            max_feat_num=input_dim_X,
            depth=num_layer,
            nhid=hidden_size
        )

    def compute_loss(self, x, adj, flags, loss_fn):
        loss_x, loss_adj = loss_fn(
            self.score_network_x, self.score_network_a, x, adj, flags
        )
        return loss_x, loss_adj
    
    def forward(self, x, adj, flags):
        raise NotImplementedError("Forward pass not implemented. Please use self.score_network_x and self.score_network_a separately.")
    
    def initialize_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

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

def get_sde_loss_fn(
    sde_x,
    sde_adj,
    train=True,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
    eps=1e-5,
):
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model_x, model_adj, x, adj, flags):
        score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn(
            sde_adj, model_adj, train=train, continuous=continuous
        )

        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)

        if not likelihood_weighting:
            losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)

            losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        else:
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = (
                reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj
            )
        return torch.mean(losses_x), torch.mean(losses_adj)

    return loss_fn
