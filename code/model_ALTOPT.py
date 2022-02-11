import torch
import torch.nn.functional as F
from torch.nn import Linear
from model import GCN
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
from torch_geometric.nn import ChebConv
import math


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, **kwargs):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, **kwargs):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, **kwargs):
        super(APPNP, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t, data=data)
        return F.log_softmax(x, dim=1)


# class ALTOPT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
#         super(ALTOPT, self).__init__()
#         self.lin1 = Linear(in_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, out_channels)
#         self.dropout = dropout
#         self.prop = prop
#         self.args = args
#         self.add_self_loops = True  # maybe false @Zhou 2003
#         self.FF = None  ## pseudo label
#         self.mlp = None
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.prop.reset_parameters()
#         self.mlp = None
#         self.FF = None
#
#     def propagate_update(self, data, K):
#         # return
#
#         # pass
#         if self.FF is None:
#             self.FF = self.prop.init_label(data)
#         # else:
#         ## maybe recompute mlp output here using the test mode
#         if self.mlp is None:
#             # print('no mlp')
#             zero_mlp = torch.zeros_like(self.FF)
#             self.FF = self.prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')
#         else:
#             # print('with mlp')
#             self.FF = self.prop(x=self.mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')
#
#     def forward(self, data):
#         x, adj_t = data.x, data.adj_t
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
#         x = F.softmax(x, dim=1)
#         ## note that the difference between training and test: dropout or not
#         if not self.training:
#             ## there is no dropout in test
#             self.mlp = x.clone().detach()
#         return x

class ALTOPT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, args, **kwargs):
        super(ALTOPT, self).__init__()
        num_layers = args.num_layers
        self.hidden_channels = hidden_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.prop = prop
        self.args = args
        self.add_self_loops = True  # maybe false @Zhou 2003
        self.FF = None  ## pseudo label
        self.mlp = None
        # torch.manual_seed(100)
        # self.reset_parameters()
        self.gcn = GCN(in_channels, hidden_channels, out_channels, dropout, 2)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.prop.reset_parameters()
        self.gcn.reset_parameters()
        self.mlp = None
        self.FF = None


    def propagate(self, data):
        x, adj_t, = data.x, data.adj_t
        self.x = self.prop(x, adj_t, data=data, mode='CS')
        if self.FF is None:
            self.FF = self.prop.init_label(data)
            # print(self.FF)

    def ensamble(self, data, out):
        x, adj_t, = data.x, data.adj_t
        out = self.prop(out, adj_t, data=data, mode='CS', alpha=0.9)
        return out

    def propagate_update(self, data, K):
        # return

        # pass
        if self.FF is None:
            self.FF = self.prop.init_label(data)
        # else:
        ## maybe recompute mlp output here using the test mode
        # import ipdb
        # ipdb.set_trace()
        if self.mlp is None:
            print('no mlp')
            zero_mlp = torch.zeros_like(self.FF)
            self.FF = self.prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')
        else:
            # print('with mlp')
            self.FF = self.prop(x=self.mlp, edge_index=data.adj_t, data=data, FF=self.FF, mode='ALTOPT')

    def forward(self, data, index=None):
        if self.FF is None:
            self.FF = self.prop.init_label(data)
        # x, adj_t = data.x, data.adj_t
        # # x = self.x
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # if index is not None:
        #     x = x[index]
        # for i, lin in enumerate(self.lins[:-1]):
        #     x = lin(x)
        #     # x = self.bns[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lins[-1](x)
        x = self.gcn(data)
        x = F.softmax(x, dim=1).clamp(min=1e-8)
        # x = F.log_softmax(x, dim=1)

        ## note that the difference between training and test: dropout or not
        if not self.training:
            ## there is no dropout in test
            self.mlp = x.clone().detach()
        return x


