from dataset import get_dataset
import argparse
import networkx as nx
import numpy as np
import torch
import os
from torch_geometric.utils.convert import to_networkx
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch_geometric
import matplotlib.pyplot as plt
# from igraph import *
from torch_geometric.utils import to_undirected, dropout_adj
from torch_sparse import SparseTensor

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(description='ALTOPT')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')
    parser.add_argument('--fix_num', type=int, default=20, help='number of train sample each class')
    parser.add_argument('--const_split', type=str2bool, default=False)
    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args


args = parse_args()
i = 0
dataset, data, split_idx = get_dataset(args, i)

# def spectral(data, post_fix):
#     print('hh')
#     from julia.api import Julia
#     jl = Julia(compiled_modules=False)
#     print('Setting spectral embedding')
#     from julia import Main
#     Main.include("./norm_spec.jl")
#     print('Setting up spectral embedding')
#
#     adj = data.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
#     print(adj.shape)
#     adj = to_undirected(adj)
#     print(adj.shape)
#     np_edge_index = np.array(adj.T)
#     N = 2708
#     row, col = adj
#     adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
#     adj = adj.to_scipy(layout='csr')
#     result = torch.tensor(Main.main(adj, 128)).float()
#     print(result)
#     torch.save(result, f'embeddings/spectral{post_fix}.pt')
#
#     return result
#
# args = parse_args()
# for i in range(10):
#     dataset, data, split_idx = get_dataset(args, i)
#     print(data.y.shape[0])
#     out = data.x
#     print(out.shape)
#     FTF = torch.mm(out.T, out)
#     print(FTF.shape)
#     FFTF = torch.mm(out, FTF)
#     print(FFTF.shape)
#     #spectral(data, 'test')
#     # g = nx.Graph()
#     # g = Graph()
#     # g.add_vertices(2708)
#     # edge = data.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
#     # u = edge[0].tolist()
#     # v = edge[1].tolist()
#     # edges = []
#     # for j in range(len(u)):
#     #     edges.append((u[j], v[j]))
#     # g.add_edges(edges)
#     # g.vs['train'] = data.train_mask.tolist()
#     # # nx.draw_networkx(g)
#     # # plt.show()
#     # layout = g.layout('large')
#     # g.vs['color'] = ['blue' if i else 'pink' for i in g.vs['train']]
#     # plot(g, layout=layout)
#     break
