# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: re_train.py
# @time: 2020/04/21
# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: train.py
# @time: 2020/03/13
import os
import time
import tqdm
import math
import itertools
import argparse
from framwork.snapshot import *
from framwork.model import *
from framwork.negative_sample import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--sample_rate', type=float, default=1, help='Sample sample_rate percent from initial edges.')
parser.add_argument('--ini_graph_percent', type=float, default=0.5, help='Train and test data percent.')
parser.add_argument('--anomaly_percent', type=float, default=0.05,
                    help='Anomaly injection with proportion of anomaly_percent.')
parser.add_argument('--snapshots_', type=int, default=5300, help='The snapshot size .')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--nmid1', type=int, default=200, help='Number of nmid1 units.')
parser.add_argument('--nmid2', type=int, default=150, help='Number of nmid2 units.')
parser.add_argument('--beta', type=float, default=1.0, help='Hyper-parameters in the score function.')
parser.add_argument('--mui', type=float, default=0.3, help='Hyper-parameters in the score function.')
parser.add_argument('--gama', type=float, default=0.6, help='Parameters in the score function.')
parser.add_argument('--w', type=int, default=3, help='Hyper-parameters in the score function.')
# parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# args.cuda = False
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# Load data
data_path = 'opsahl-ucsocial/out.opsahl-ucsocial'
Net1 = GCN(nfeat=args.hidden, nmid1=args.nmid1, nmid2=args.nmid2, nhid=args.hidden, dropout=args.dropout)
Net2 = HCA(hidden=args.hidden, dropout=args.dropout)
Net3 = GRU(hidden=args.hidden, dropout=args.dropout)
Net4 = Score(beta=args.beta, mui=args.mui, hidden=args.hidden, dropout=args.dropout)
N_S = negative_sample()
# optimizer = optim.Adam(itertools.chain(Net1.parameters(), Net2.parameters(), Net3.parameters(), Net4.parameters()),
#                        lr=args.lr,
#                        )  # weight_decay=args.weight_decay
optimizer = optim.Adam(itertools.chain(Net1.parameters(), Net2.parameters(), Net3.parameters(), Net4.parameters()),
                      lr=args.lr,
                      )  # weight_decay=args.weight_decay
# snapshots_train, l_train, snapshots_test, l_test, nodes, n_train = snapshot(data_path=data_path, sample_rate=args.sample_rate,
#                                                                    ini_graph_percent=args.ini_graph_percent,
#                                                                    anomaly_percent=args.anomaly_percent,
#                                                                    snapshots_=args.snapshots_)
# np.savez("snapshot_5a53.npz",snapshots_train = snapshots_train, l_train = l_train, snapshots_test = snapshots_test, l_test = l_test, nodes = nodes, n_train = n_train)
snapshots=np.load("snapshot_5a53.npz", allow_pickle=True)
snapshots_train, l_train, nodes, n_train=snapshots['snapshots_train'], snapshots['l_train'], snapshots['nodes'], snapshots['n_train']
l_train = int(l_train)
nodes = int(nodes)
n_train = int(n_train)
# args.cuda = torch.cuda.is_available()

print('===> Loding models...')
# dir_ = './checkpoint/adj.pth'
# checkpoint_ = torch.load(dir_)
# adj = checkpoint_['adj']
# print('epoch{}'.format(0))
# dir = './checkpoint/NEWR_Adam_lr_0.001_w_3_epoch{}.pth'.format(16)
# dir = './checkpoint/S5_NEW_Adam_lr_0.001_w_3_epoch{}.pth'.format(30)
dir = './checkpoint/S53_Adam_lr_0.001_w_3_epoch{}.pth'.format(21)
checkpoint = torch.load(dir)
Net1.load_state_dict(checkpoint['Net1'])
Net2.load_state_dict(checkpoint['Net2'])
Net3.load_state_dict(checkpoint['Net3'])
Net4.load_state_dict(checkpoint['Net4'])
# H_list = checkpoint['H_list']

if args.cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('OK')
else:
    torch.set_default_tensor_type(torch.FloatTensor)


def train():
    # torch.set_default_tensor_type(torch.FloatTensor)
    # CUDA_VISIBLE_DEVICES = 0
    t = time.time()
    Net1.train()
    Net2.train()
    Net3.train()
    Net4.train()
    N_S.train()
    # H = torch.zeros(nodes, args.hidden)n
    # H_ = torch.zeros(3, nodes, args.hidden)
    # H_list = torch.zeros(1, nodes, args.hidden)
    # H_=torch.zeros((args.w, nodes, args.hidden))
    # for i in range(args.w-1):
    #     H_list=torch.cat([H_list, torch.zeros(nodes, args.hidden).unsqueeze(0)], dim=0)
    # H_list.append(torch.zeros((nodes, args.hidden)))
    # H_list[args.w-1]=torch.randn(nodes, args.hidden)

    # optimizer.zero_grad()
    for epoch in range(args.epochs):
        # snapshots_train = snapshots_train.cuda()
        H_list = torch.zeros(1, nodes, args.hidden)
        H_ = torch.zeros((args.w, nodes, args.hidden))
        for k in range(args.w - 1):
            H_list = torch.cat([H_list, torch.zeros(nodes, args.hidden).unsqueeze(0)], dim=0)
        # stdv = 1. / math.sqrt(H_list[-1].size(1))
        # H_list[-1].data.uniform_(-stdv, stdv)
        nn.init.sparse_(H_list[-1][:n_train, :], sparsity=0.9)
        adj = torch.zeros((nodes, nodes))
        loss_a = torch.zeros(1)
        for i in range(l_train):
            optimizer.zero_grad()
            # snapshot=snapshots_train[i]
            snapshot = torch.from_numpy(snapshots_train[i])
            H = H_list[-1]
            for j in range(args.w):
                H_[j] = H_list[-args.w + j]
            adj, Adj = update_adj(adj=adj, snapshot=snapshot, nodes=nodes)
            Adjn = normalize_adj(Adj + torch.eye(Adj.shape[0]))
            # adj_ = torch.from_numpy(adjn)
            if args.cuda:
                Net1.cuda()
                Net2.cuda()
                Net3.cuda()
                Net4.cuda()
                N_S.cuda()
                H = H.cuda()
                Adjn = Adjn.cuda()
                H_ = H_.cuda()
                snapshot = snapshot.cuda()
            # print(n_loss)
            # print(n_loss.shape)
            H, H_, Adjn, snapshot = Variable(H), Variable(H_), Variable(Adjn), Variable(snapshot)
            current = Net1(x=H, adj=Adjn, Adj=Adj)
            short = Net2(C=H_)
            Hn = Net3(current=current, short=short)
            H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)
            # H_list.append(Hn)
            # n_data,n_loss=negative_sample(adj=adj, snapshot=snapshot, H=Hn, f=Net4)
            n_loss = N_S(adj=adj, Adj=Adj, snapshot=snapshot, H=Hn, f=Net4, arg=args.cuda)
            loss1 = args.weight_decay * (Net1.loss() + Net2.loss() + Net3.loss() + Net4.loss())
            lens = n_loss.shape[0]
            zero = torch.zeros(1)
            loss2 = torch.zeros(1)
            for m in range(lens):
                count = n_loss[m]
                loss2 = loss2 + torch.where((args.gama + count) >= 0, (args.gama + count), zero)
            loss_a = loss_a + loss1 / (l_train) + loss2 / (l_train * lens)
            # loss2=sum(torch.where((args.gama+n_loss)>=0, (args.gama+n_loss),zero))
            # loss2=args.gama+n_loss
            loss = loss1 + loss2
            # torch.autograd.set_detect_anomaly(True)
            # loss.backward()
            # torch.autograd.backward(loss2, retain_graph=True)
            loss.backward()
            optimizer.step()
            print(i)
            print('Loss of {}'.format(epoch), 'epoch,{}'.format(i), 'snapshot,loss:{}'.format(loss.item()))
            # with SummaryWriter(comment='Net1') as w:
            #     w.add_graph(Net1, (H, Adjn, Adj))
            # with SummaryWriter(comment='Net2') as w:
            #     w.add_graph(Net2, (H_,))
            # with SummaryWriter(comment='Net3') as w:
            #     w.add_graph(Net3, (current, short))
            # with SummaryWriter(comment='Net4') as w:
            #     w.add_graph(Net4, (H[0], H[1]))
            # print(loss.device)
            # print(Hn.type())
            # print(Hn)
        print('The average loss of {}'.format(epoch), 'epoch is :{}'.format(loss_a.item()))
        print(time.time() - t)
        writer1 = SummaryWriter('runs/R_Adam_w3_loss')
        # writer1 = SummaryWriter('runs/Adam_w3_loss')
        writer1.add_scalar('loss_avarage', loss_a.item(), epoch)
        print('===> Saving models...')
        state = {'Net1': Net1.state_dict(), 'Net2':
            Net2.state_dict(), 'Net3': Net3.state_dict(),
                 'Net4': Net4.state_dict(),
                'H_list': H_list, 'loss_a': loss_a, 'epoch': epoch}
        # print(n_loss)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        dir = './checkpoint/S53_Adam_lr_0.001_w_3_epoch{}.pth'.format(epoch)
        torch.save(state, dir)


    adj = {'adj': adj}
    dir = './checkpoint/adj.pth'
    torch.save(adj, dir)
    print('Finish')
    # return adj, H_list

    # print(type(Net1.state_dict()))  # 查看state_dict所返回的类型，是一个“顺序字典OrderedDict”
    # print(type(Net2.state_dict()))
    # print(type(Net3.state_dict()))
    # print(type(Net4.state_dict()))
    # print(type(N_S.state_dict()))
    #
    # for param_tensor in N_S.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    # print(param_tensor, '\t', N_S.state_dict()[param_tensor].size())
    # for var_name in optimizer.state_dict():
    #     print(var_name,'\t',optimizer.state_dict()[var_name])


# print(snapshots_test[0])

if __name__ == "__main__":
    train()
