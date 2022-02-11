from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor, matmul
import numpy as np

from myutil import set_signal_by_label  ## remove

def get_inc(edge_index):
    # compute the incident matrix
    size = edge_index.sizes()[1]
    row_index = edge_index.storage.row()
    col_index = edge_index.storage.col()
    mask = row_index >= col_index # remove duplicate edge and self loop
    row_index = row_index[mask]
    col_index = col_index[mask]
    edge_num = row_index.numel()
    row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
    col = torch.cat([row_index, col_index])
    value = torch.cat([torch.ones(edge_num), -1*torch.ones(edge_num)]).cuda()
    inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                        sparse_sizes=(edge_num, size))
    return inc

def inc_norm(inc, edge_index):
    ## edge_index: unnormalized adjacent matrix
    ## normalize the incident matrix
    edge_index = torch_sparse.fill_diag(edge_index, 1.0) ## add self loop to avoid 0 degree node
    deg = torch_sparse.sum(edge_index, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    inc = torch_sparse.mul(inc, deg_inv_sqrt.view(1, -1)) ## col-wise
    return inc

def check_inc(edge_index, inc):
    nnz = edge_index.nnz()
    deg = torch.eye(edge_index.sizes()[0]).cuda()
    adj = edge_index.to_dense()
    lap = (inc.t() @ inc).to_dense()
    lap2 = deg - adj
    diff = torch.sum(torch.abs(lap2-lap)) / nnz
    assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'


class EMP(MessagePassing):
    r"""The elastive message passing layer from 
    the paper "Elastic Graph Neural Networks", ICML 2021
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]
    _cached_inc = Optional[SparseTensor]

    def __init__(self, 
                 K: int, 
                 mode: str,
                 lambda1: float = None,
                 lambda2: float = None,
                 alpha: float = None,
                 L21: bool = True,
                 dropout: float = 0,
                 cached: bool = False, 
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 args = None,
                 **kwargs):

        super(EMP, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.mode = mode
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        
        assert add_self_loops == True and normalize == True, ''
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None  ## incident matrix

        self.args = args

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None


    def forward(self, x: Tensor, 
                edge_index: Adj, 
                edge_weight: OptTensor = None, 
                data=None) -> Tensor:
        """"""
        if self.K <= 0: return x

        assert isinstance(edge_index, SparseTensor), "Only support SparseTensor now"
        assert edge_weight is None, "edge_weight is not supported yet, but it can be extented to weighted case"

        self.unnormalized_edge_index = edge_index

        if self.normalize:
            cache = self._cached_inc
            if cache is None:
                inc_mat = get_inc(edge_index=edge_index)               ## compute incident matrix before normalizing edge_index
                inc_mat = inc_norm(inc=inc_mat, edge_index=edge_index) ## normalize incident matrix

                if self.cached:
                    self._cached_inc = inc_mat
                    self.init_z = torch.zeros((inc_mat.sizes()[0], x.size()[-1])).cuda()
            else:
                inc_mat = self._cached_inc

            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)

                if x.size()[0] < 30000:
                    check_inc(edge_index=edge_index, inc=inc_mat) ## ensure L=B^TB

                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        hh = x

        ## to remove
        if self.args.LP:
            x = set_signal_by_label(x, data)
        ## to remove
        # self.test_pattern(x=x, hh=hh, k=self.K, data=data, edge_index=edge_index, inc=inc_mat)

        if self.mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K, alpha=self.alpha)
        elif self.mode == 'EMP':
            x = self.emp_forward(x=x, hh=hh, edge_index=edge_index, inc=inc_mat, K=self.K)
        elif self.mode == 'CP':
            x = self.cp_forward(x=x, hh=hh, edge_index=edge_index, inc=inc_mat, K=self.K)
        else:
            raise ValueError('wrong propagate mode')

        return x

    ## to remove
    def appnp_forward(self, x, hh, edge_index, K, alpha):

        # if self.training:
        #     ## MLP training, APPNP test
        #     K = 0
        # if not self.training:
        #     ## APPNP training, MLP test
        #     K = 0
        # K = 0 # MLP training and MLP test

        for k in range(K):
            x = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = x * (1 - alpha)
            x += alpha * hh
            # print('hhh')
            x = F.dropout(x, p=self.dropout, training=self.training)
        # import ipdb; ipdb.set_trace()
        return x

    def emp_forward(self, x, hh, K, edge_index, inc):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma   = 1/(1+lambda2)
        beta = 1/(2*gamma)

        if lambda1 > 0: 
            z = self.init_z.detach()

        for k in range(K):

            if lambda2 > 0:
                # y = x - gamma * (x - hh + lambda2 * (x - self.propagate(edge_index, x=x, edge_weight=None, size=None)))
                ## simplied as the following if gamma = 1/(1+lambda2):
                y = gamma * hh + (1-gamma) * self.propagate(edge_index, x=x, edge_weight=None, size=None)
            else:
                y = gamma * hh + (1-gamma) * x # y = x - gamma * (x - hh)

            if lambda1 > 0:
                x_bar = y - gamma * (inc.t() @ z)
                z_bar  = z + beta * (inc @ x_bar)
                if self.L21:
                    z  = self.L21_projection(z_bar, lambda_=lambda1)
                else:
                    z  = self.L1_projection(z_bar, lambda_=lambda1)
                x = y - gamma * (inc.t() @ z)
            else:
                x = y # z=0

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def L1_projection(self, x: Tensor, lambda_):
        return torch.clamp(x, min=-lambda_, max=lambda_)
    
    def L21_projection(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]
        return scale.unsqueeze(1) * x

    def cp_forward(self, x, hh, K, edge_index, inc):
        alpha = self.alpha
        gamma = 1 ## any positive number
        # gamma = 0.5 ## any positive number
        beta  = 1/(2*gamma)
        # beta  = 1/(2*gamma+0.1)
        # beta  = 1/(4*gamma)

        z = self.init_z.detach()
        x_tilde = x

        for k in range(K):
            z_bar = z + beta * (inc @ x_tilde)
            z     = self.L21_projection(z_bar, lambda_=1-alpha)
            x_bar = x - gamma * (inc.t() @ z)
            x_old = x
            x     = self.CP_L21_proximal(x_bar, beta*alpha)
            x_tilde = 2 * x - x_old
            # loss = self.CP_objective(x=x, hh=hh, alpha=alpha, inc=inc)
            # print(f'step = {k},  loss: {loss}')
        # import ipdb; ipdb.set_trace()
        return x ## or x_tilde??

    def CP_objective(self, x, hh, alpha, inc):
        # import ipdb; ipdb.set_trace()
        feature_loss = torch.norm(x-hh, p=2, dim=1).sum()
        graph_loss   = torch.norm(inc @ x, p=2, dim=1).sum()
        return alpha*feature_loss + (1-alpha)*graph_loss

    def CP_L21_proximal(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm-lambda_, min=0)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]
        return scale.unsqueeze(1) * x

    # def CP_L21_proximal_conjugate(self, x: Tensor, lambda_):
    ## the same as L21_projection
    #     row_norm = torch.norm(x, p=2, dim=1)
    #     scale = torch.clamp(row_norm, max=lambda_)
    #     index = row_norm > 0
    #     scale[index] = scale[index] / row_norm[index]
    #     return scale.unsqueeze(1) * x

    ## to remove
    def test_pattern(self, x, hh, k, data, edge_index, inc):
        if not self.training and (self.args.current_epoch+1) % 200 == 0: ## no dropout in testing
            
            print('before propagation')
            self.see_pattern(x=x, hh=hh, k=k, data=data)

            print('APPNP propagation')
            y = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=k, alpha=0.1) # 0.1
            self.see_pattern(x=y, hh=hh, k=k, data=data)

            print('L1 EMP propagation')
            self.L21 = False
            y = self.emp_forward(x=x, hh=hh, edge_index=edge_index, inc=inc, K=k)
            self.see_pattern(x=y, hh=hh, k=k, data=data)

            print('L21 EMP propagation')
            self.L21 = True
            y = self.emp_forward(x=x, hh=hh, edge_index=edge_index, inc=inc, K=k)
            self.see_pattern(x=y, hh=hh, k=k, data=data)

            
            import ipdb; ipdb.set_trace()

    ## to remove
    def see_pattern(self, x, hh, data=None, k=None):
        # edge_index = self._cached_adj_t
        edge_index = self.unnormalized_edge_index  ## can we use normalize one?
        inc_mat = self._cached_inc
        cx  = inc_mat @ x

        ## seperate edge: right edge / wrong edge
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        mask = row_index >= col_index
        # import ipdb; ipdb.set_trace()
        row_index = row_index[mask]
        col_index = col_index[mask]
        label = data.y
        correct_link = label[row_index] == label[col_index]
        row_norm = torch.norm(cx, p=2, dim=1)
        print('row_norm sort: ', torch.sort(row_norm)[0])
        row_norm_correct = row_norm[correct_link]
        row_norm_wrong = row_norm[~correct_link]
        print('correct_link node diff mean: ', row_norm_correct.mean())
        print('wrong_link node diff mean: ', row_norm_wrong.mean())
        print('ratio: ', row_norm_correct.mean() / row_norm_wrong.mean() )
        print('inverse ratio: ', row_norm_wrong.mean() / row_norm_correct.mean() )

        sort_norm = torch.sort(row_norm)[0]
        # sparse_ratio = (sort_norm<0.1).sum().item() / sort_norm.nelement()
        # print('sparse_ratio: <0.1   ', sparse_ratio)
        thresh = 0.1
        sparse_ratio = (sort_norm<0.1).sum().item() / sort_norm.nelement()
        print(f'sparse_ratio:  < {thresh}   ', sparse_ratio)
        import ipdb; ipdb.set_trace()

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, mode={}, lambda1={}, lambda2={}, L21={}, alpha={})'.format(
            self.__class__.__name__, self.K, self.mode, self.lambda1, self.lambda2, self.L21, self.alpha)
