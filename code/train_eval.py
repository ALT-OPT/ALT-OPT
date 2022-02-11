import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
import  math
import torch.nn as nn

def cross_entropy(pred, target):
    pred = torch.log(pred)
    return torch.mean(torch.sum(-target * pred, 1))

def cross_entropy1(pred, target):
    # pred = pred.clamp(min=1e-8)
    pred = torch.log_softmax(pred, dim=-1)
    # pred = torch.log(pred)
    return -torch.sum(target * pred, 1)

def KL(pred, target):
    return F.kl_div(pred.log(), target)


def train_altopt_PTA(model, data, train_idx, optimizer, args=None):
    model.train()
    label = model.FF
    train_mask = data.train_mask
    optimizer.zero_grad()
    y_hat = model(data=data)
    # import ipdb
    # ipdb.set_trace()
    gamma = math.log(1 + (args.current_epoch-1)/100)
    y_hat_con = torch.detach(torch.softmax(y_hat, dim=-1))
    # loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(label, y_hat_con ** gamma))) / args.num_class  # PTA
    # loss = - torch.sum(
        # torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(label, y_hat_con))) / args.num_class
    loss = - torch.sum(
        torch.mul(torch.log_softmax(y_hat, dim=-1), label)) / args.num_class  # PTA
    # out = torch.softmax(y_hat, dim=-1)
    # out1 = out.clone().detach() ** gamma
    # label = label * out1
    # loss = cross_entropy1(out, label)
    # loss = loss.sum()
    # loss = torch.sum(weight * loss)
    loss.backward()
    optimizer.step()
    # print(loss.item())
    return loss.item()



def train_altopt(model, data, train_idx, optimizer, args=None):
    # print('train')
    y = data.y
    # model.eval()
    # out1 = model(data=data)
    model.train()
    # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    optimizer.zero_grad()
    out = model(data=data) #[train_idx]
    label = model.FF
    '''
    mlp acc
    '''
    # evaluator = Evaluator(name='ogbn-arxiv')
    # test_mask = data.test_mask
    # mlp_test = out1[test_mask].argmax(dim=1, keepdim=True)
    # y_test = y[test_mask].unsqueeze(dim=1)
    # test_acc = evaluator.eval({
    #     'y_true': y_test,
    #     'y_pred': mlp_test,
    # })['acc']
    # print('test_acc', test_acc)




    if label is None:
        label = model.prop.init_label(data)
    train_mask = data.train_mask
    pseudo_mask = data.pseudo_mask

    '''
    pseudo label const each class
    add to train dataset
    '''
    # if args.current_epoch < 100:
    #     alpha = 0
    # elif args.current_epoch < 600:
    #     alpha = 0.001 * args.current_epoch
    # alpha = 0.3 * (args.current_epoch - 1)
    # if args.current_epoch % 100 == 0:
    #     num = 100
    #     for i in range(args.num_class):
    #         weight = 1 - torch.sum(-label * torch.log(label), 1) / math.log(args.num_class)
    #         _, index = label.max(dim=1)
    #         pos = (index == i)
    #         weight[~pos] = 0
    #         weight[pseudo_mask] = 0
    #         value, indices = torch.topk(weight, num)
    #         data.pseudo_mask[indices] = 1
    #         data.pseudo_label[indices] = i
    #     # print(data.pseudo_mask.sum())
    # loss = cross_entropy(out[pseudo_mask], label[pseudo_mask])
    # import ipdb
    # ipdb.set_trace()
    total_weight = torch.zeros(y.shape[0]).cuda()
    # total_weight = data.total_weight
    total_weight[train_mask] = 1

    # num = args.current_epoch // 100 * 100
    # total_weight = data.total_weight


    if args.current_epoch == 0:
        num = 0
    elif args.current_epoch == 1:
        num = 100
    else:
        num = 100
    # num = 100
    y_train = y[train_mask]
    class_num = torch.zeros(args.num_class).cuda()
    real_distribution = []
    test_distribution = []
    # for i in range(args.num_class):
        # real_distribution.append(sum(data.y==i).item())
    # # if args.current_epoch == 1:
    # import ipdb
    # ipdb.set_trace()
    # if True:
    if args.current_epoch > 0:
        for i in range(args.num_class):
            weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
            _, index = label.max(dim=1)
            pos = (index == i)
            # test_distribution.append(sum(pos).item())
            weight[~pos] = 0
            weight[train_mask] = 0
            value, indices = torch.topk(weight, num)
            total_weight[indices] = value
            # indices = weight.nonzero(as_tuple=True)[0]
            # import ipdb; ipdb.set_trace()
            # value = weight[indices]
            # class_num[i] = 300 / (indices.shape[0] + 5)
            # total_weight[indices] = value * class_num[i]
        # total_weight[train_mask] = class_num[y_train]


        # import ipdb; ipdb.set_trace()


        psuedo_indices = total_weight.nonzero().squeeze()
        # if args.current_epoch == 1:
        #     torch.save(data, 'data.th')
        #     torch.save(total_weight, 'total_weigh.th')
        #     torch.save(label, 'label.th')
        #     torch.save(out, 'out.th')
        #     data.psuedo_indices = psuedo_indices

        """
            Test the accuracy of psuedo label
        """

        # import ipdb
        # ipdb.set_trace()
        # if args.current_epoch > 0:
        #     psuedo_indices = data.psuedo_indices
        #     y = data.y
        #     true_label = y[psuedo_indices].unsqueeze(dim=1)
        #     psuedo_label = label.argmax(dim=1, keepdim=True)[psuedo_indices]
        #     evaluator = Evaluator(name='ogbn-arxiv')
        #     if len(psuedo_indices) > 0:
        #         psuedo_acc = evaluator.eval({
        #             'y_true': true_label,
        #             'y_pred': psuedo_label,
        #         })['acc']
        #         print('psuedo_acc', psuedo_acc, args.current_epoch)

        # total_weight[train_mask] = 1
        # data.total_weight = total_weight



        # print(total_weight.nonzero().squeeze().shape)
    # test_portion = []
    # for i in range(args.num_class):
    #     test_portion.append(test_distribution[i]/real_distribution[i])
    # print('test class', test_portion)

    # total_weight[train_mask] = 1
    # diff = cross_entropy1(out, label)
    # diff = torch.softmax(out, dim=-1) - label
    # all_train_index = total_weight.nonzero().squeeze()
    # import ipdb
    # ipdb.set_trace()
    # diff = diff[all_train_index]
    # diff = torch.sum(diff * diff, 1)
    # loss = diff.sum()
    diff = out - label
    diff = torch.sum(diff * diff, 1)
    loss = torch.sum(total_weight * diff)
    # loss = loss

    """
    batch train
    """
    batch_size = 100
    # all_train_index = total_weight.nonzero().squeeze()
    #
    # optimizer.zero_grad()
    # out = model(data)[all_train_index]
    # # out = model(data, all_train_index)
    # diff = out - label[all_train_index]
    # diff = torch.sum(diff * diff, 1)
    # loss = diff.sum()
    # loss.backward()
    # optimizer.step()

    # print(len(all_train_index))
    # import ipdb
    # ipdb.set_trace()
    # idx = torch.randperm(all_train_index.shape[0])
    # all_train_index = all_train_index[idx].split(batch_size)
    # # all_train_index = all_train_index.split(batch_size)
    # for index in all_train_index:
    #     # o = model(data, index)
    #     o = model(data, index)
    #     diff = o - label[index]
    #     loss = torch.sum(diff * diff)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # print(args.current_epoch)

    """
    deep SSL
    """
    # if args.current_epoch == 0:
    #     loss = F.nll_loss(out[train_mask].log(), y[train_mask])
    # else:
    #     weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
    #     y_train = y[train_mask]
    #     y_psuedo = label[~train_mask].argmax(dim=1)
    #     class_num = torch.zeros(args.num_class).cuda()
    #     for i in range(args.num_class):
    #         n = sum(y_train == i)
    #         n += sum(y_psuedo == i)
    #         class_num[i] = 1/n
    #     y_train_weight = class_num[y_train]
    #     y_psuedo_weight = class_num[y_psuedo]
    #     label_loss = y_train_weight * F.nll_loss(out[train_mask].log(), y_train, reduction='none')
    #     psuedo_loss = y_psuedo_weight * weight[~train_mask] * F.nll_loss(out[~train_mask].log(), y_psuedo, reduction='none')
    #     loss = label_loss.sum() + psuedo_loss.sum()

    """
    true train
    """
    # y = data.y[train_mask]
    # out = torch.log(out[train_mask])
    # # out = F.log_softmax(out[train_mask], dim=-1)
    # loss = F.nll_loss(out, y)

    # if args.loss == 'CE':
    #     # out = F.log_softmax(out, dim=1)
    #     # label_loss = cross_entropy(out[train_mask], label[train_mask])
    #     label_loss = F.nll_loss(out[train_mask].log(), y[train_mask])
    #     kl_loss = KL(out, label)
    #     loss = label_loss + 0.2*kl_loss
    #     # print('label', label[train_mask][0])
    #     # print('out', out[train_mask][0])
    #     # unlabel_loss = cross_entropy(out[~train_mask], label[~train_mask])
    # elif args.loss == 'MSE':
    #     # diff = out - label
    #     # label_loss = torch.pow(torch.norm(diff[train_mask]), 2)
    #     # unlabel_loss = torch.pow(torch.norm(diff[~train_mask]), 2)
    #     label_loss = F.nll_loss(out[train_mask].log(), y[train_mask])
    #     kl_loss = KL(out, label)
    #     loss = label_loss + kl_loss
    # elif args.loss == 'CEM':
    #     # out = F.log_softmax(out, dim=1)
    #     out = out.clamp(min=1e-6)
    #     out = torch.log(out)
    #     y = label.argmax(dim=1)
    #     label_loss = F.nll_loss(out[train_mask], y[train_mask])
    #     unlabel_loss = F.nll_loss(out[~train_mask], y[~train_mask])
    # # args.lambda1 = math.log(1+args.current_epoch/1000)
    # weight = 0
    # if args.has_weight:
    #     loss = label_loss + weight * unlabel_loss
    # else:
    #     loss = label_loss + unlabel_loss

    # if len(data.y.shape) == 1:
    #     y = data.y[train_idx]
    # else:
    #     y = data.y.squeeze(1) #[train_idx]  ## for ogb data
    # label = model.FF

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_altopt(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    if args.model == 'ALTOPT':
        out = model(data=data)  ## still forward to update mlp output, or move it to propagate_update
        # out = torch.softmax(out, dim=-1)
        out = model.FF ## hidden
        # out = model.ensamble(data, out)
    else:
        out = model(data=data)
    
    # out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    # import ipdb; ipdb.set_trace()

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))

    # print(f'train_acc: {train_acc}, valid_acc: {valid_acc}, test_acc: {test_acc}')
    return train_acc, valid_acc, test_acc
    # return -train_loss, -valid_loss, -test_loss

def train(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out) 
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out-label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))
    
    return train_acc, valid_acc, test_acc
    # return out, train_acc, valid_acc, test_acc
    # return -train_loss, -valid_loss, -test_loss


def train_appnp(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    torch.autograd.set_detect_anomaly(True)  ## to locate error of NaN
    optimizer.zero_grad()
    out, loss1 = model(data=data)
    out = out[train_idx]
    # out = model(data=data)[train_idx]

    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        if args.current_epoch > 100:
            loss = F.nll_loss(out, y) + 0.005 * loss1
            # loss = F.nll_loss(out, y)
        else:
            loss = F.nll_loss(out, y)
        # loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out)
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out - label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_appnp(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out, diff = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test1(model, data, out, split_idx, args=None):
    # print('test')
    model.eval()
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    print('base model test_acc', test_acc)
    return train_acc, valid_acc, test_acc


def train_cs(model, data, train_idx, optimizer, args=None):
    model.train()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    out = F.log_softmax(out, dim=-1)
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    loss = F.nll_loss(out, y)
    # loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_cs(model, data, split_idx, out=None, args=None):
    model.eval()
    if out is None:
        out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc, out
