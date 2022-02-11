from dataset import get_dataset
import argparse
from util import str2bool
import networkx as nx
from prop import Propagation
import torch
import math
from ogb.nodeproppred import Evaluator
from collections import defaultdict, Counter

def parse_args():
    parser = argparse.ArgumentParser(description='ALTOPT')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--fix_num', type=int, default=20, help='number of train sample each class')
    parser.add_argument('--proportion', type=float, default=0, help='proportion of train sample each class')
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=1, help='default use fix split')
    parser.add_argument('--prop', type=str, default='EMP')
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=5)
    parser.add_argument('--loss', type=str, default='CE', help='CE, MSE')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args

def get_graph(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    adj = data.adj_t.to_torch_sparse_coo_tensor().coalesce().indices()
    for u, v in zip(*adj.tolist()):
        G.add_edge(u, v)
    return G


args = parse_args()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

def run_path(data=None, FF=None, out=None):
    if data is None:
        dataset, data, split_idx = get_dataset(args, 0, defense=None)
    g = get_graph(data)
    data = data.to(device)
    args.num_class = data.y.max()+1
    args.current_epoch = 1
    train_mask = data.train_mask
    test_mask = data.test_mask

    prop = Propagation(K=args.K,
                       alpha=args.alpha,
                       mode=args.prop,
                       cached=True,
                       args=args)
    FF1 = None
    # if FF is None:
    if True:
        FF1 = prop.init_label(data)
        zero_mlp = torch.zeros_like(FF1)
        FF1 = prop(x=zero_mlp, edge_index=data.adj_t, data=data, FF=FF1, mode='ALTOPT')
        # print(FF1)
    label = FF
    y = data.y
    total_weight = torch.zeros(y.shape[0]).cuda()
    num = 200
    for i in range(args.num_class):
        weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
        _, index = label.max(dim=1)
        pos = (index == i)
        weight[~pos] = -10000
        weight[train_mask] = -10000
        # weight = -1 * weight
        value, indices = torch.topk(weight, num)
        total_weight[indices] = 1
    psuedo_indices = total_weight.nonzero().squeeze()
    print('psuedo_indices:', len(psuedo_indices))
    # import ipdb
    # ipdb.set_trace()
    true_label = y[psuedo_indices].unsqueeze(dim=1)
    psuedo_label = label.argmax(dim=1, keepdim=True)[psuedo_indices]
    evaluator = Evaluator(name='ogbn-arxiv')
    if len(psuedo_indices) > 0:
        psuedo_acc = evaluator.eval({
            'y_true': true_label,
            'y_pred': psuedo_label,
        })['acc']
    print('psuedo_acc', psuedo_acc)
    test_label = label.argmax(dim=1, keepdim=True)[test_mask]
    test_true_label = y[test_mask].unsqueeze(dim=1)

    train_acc = evaluator.eval({
        'y_true': test_true_label,
        'y_pred': test_label,
    })['acc']
    print('test_acc', train_acc)
    '''
    Why some node are wrong?
    '''
    test_wrong_index = (test_label != test_true_label).squeeze().nonzero(as_tuple=True)[0]
    # mlp_test = out[test_wrong_index].argmax(dim=1, keepdim=True)
    # FF1_test = FF1[test_wrong_index].argmax(dim=1, keepdim=True)
    mlp_test = out[test_mask].argmax(dim=1, keepdim=True)
    FF1_test = FF1[test_mask].argmax(dim=1, keepdim=True)
    # test_wrong_label = test_true_label[test_wrong_index]
    mlp_test_acc = evaluator.eval({
            # 'y_true': test_wrong_label,
            'y_true': test_true_label,
            'y_pred': mlp_test,
        })['acc']
    # mlp_correct_index = (mlp_test == test_wrong_label).squeeze().nonzero(as_tuple=True)[0]
    # FF1_correct_index = (FF1_test == test_wrong_label).squeeze().nonzero(as_tuple=True)[0]
    # print(mlp_correct_index)
    # print(FF1_correct_index)
    print('mlp_test_acc', mlp_test_acc)
    FF1_test_acc = evaluator.eval({
            # 'y_true': test_wrong_label,
            'y_true': test_true_label,
            'y_pred': FF1_test,
        })['acc']
    print('FF1_test_acc', FF1_test_acc)

    '''
    The accuracy of label prop = MLP
    '''
    mlp_test = out[test_mask].argmax(dim=1, keepdim=True)
    FF1_test = FF1[test_mask].argmax(dim=1, keepdim=True)
    same_index = (mlp_test == FF1_test).squeeze().nonzero(as_tuple=True)[0]
    true_test = test_true_label[same_index]
    same_test = mlp_test[same_index]
    acc = evaluator.eval({
            'y_true': true_test,
            # 'y_true': test_true_label,
            'y_pred': same_test,
        })['acc']
    print('same_acc', acc)
    import ipdb
    ipdb.set_trace()
    if out is not None:
        mlp_label = out[psuedo_indices].argmax(dim=1, keepdim=True)
        # mlp_label = out[test_mask].argmax(dim=1, keepdim=True)
        mlp_acc = evaluator.eval({
            'y_true': true_label,
            # 'y_true': test_true_label,
            'y_pred': mlp_label,
        })['acc']
        print('mlp_acc:', mlp_acc)
    if FF1 is not None:
        FF1_label = FF1[psuedo_indices].argmax(dim=1, keepdim=True)
        # FF1_label = FF1[test_mask].argmax(dim=1, keepdim=True)
        prop_acc = evaluator.eval({
            'y_true': true_label,
            # 'y_true': test_true_label,
            'y_pred': FF1_label,
        })['acc']
        print('prop_acc:', prop_acc)
        wrong_index = (FF1_label != true_label).squeeze().nonzero(as_tuple=True)[0]
        # import ipdb
        # ipdb.set_trace()
        # wrong_index = (FF1_label == true_label).squeeze().nonzero(as_tuple=True)[0]
        wrong_index = psuedo_indices[wrong_index]
        mlp_wrong_index = out[wrong_index].argmax(dim=1, keepdim=True)
        true_wrong_index = y[wrong_index].unsqueeze(dim=1)

        mlp_wrong_acc = (mlp_wrong_index == true_wrong_index).squeeze().nonzero().shape[0] / wrong_index.shape[0]
        print('mlp_wrong_acc', mlp_wrong_acc)
        psuedo_wrong_index = label[wrong_index].argmax(dim=1, keepdim=True)
        psuedo_wrong_acc = (psuedo_wrong_index == true_wrong_index).squeeze().nonzero().shape[0] / wrong_index.shape[0]
        print('psuedo_wrong_acc', psuedo_wrong_acc)



    train_indices = train_mask.nonzero().squeeze()
    psuedo_indices = psuedo_indices.cpu().tolist()
    target = train_indices.cpu().tolist()

    class2nodes = defaultdict(list)
    for i in target:
        c = y[i].item()
        class2nodes[c].append(i)

    paths = defaultdict(list)
    for i in psuedo_indices:
        for t in target:
            try:
                d = nx.shortest_path(g, source=i, target=t)
            except:
                continue
            paths[i].append(d)
    # import ipdb
    # ipdb.set_trace()
    print('paths', len(paths))
    shortests = []
    means = []
    st = {}
    for n, path in paths.items():
        shortest = 100
        s_nodes = -1
        path_lens = []
        for p in path:
            s = len(p) - 1
            path_lens.append(s)
            if s < shortest:
                shortest = s
                s_nodes = p[-1]
        shortests.append(shortest)
        means.append(sum(path_lens)/len(path_lens))
        st[n] = s_nodes
    print('shortests', shortests)
    print(Counter(shortests).most_common())

    cnum = 0
    wrong_num = 0
    true_length = []
    print('st:', len(st))
    for k, v in st.items():
        plabel = label[k].argmax().item()
        tplabel = y[k].item()
        tlabel = y[v].item()
        if plabel == tlabel:
            cnum += 1
        if plabel != tplabel:
            # print(tplabel, label[k])
            wrong_num += 1
            lens = []
            for p in paths[k]:
                if p[-1] in class2nodes[tplabel]:
                    lens.append(len(p))
            if len(lens) > 0:
                true_length.append(min(lens))
    print('wrong_num:', wrong_num)
    print(true_length)
    print(Counter(true_length).most_common())
    print(cnum/len(st))


if __name__ == '__main__':
    data = torch.load('data.th')
    FF = torch.load('label.th')
    out = torch.load('out.th')
    # import ipdb
    # ipdb.set_trace()
    run_path(data, FF, out)
