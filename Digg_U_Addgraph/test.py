# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: test.py
# @time: 2020/03/20
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from framwork.snapshot import *
from framwork.model import *
from framwork.negative_sample import update_adj
from AUC import AUC

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--sample_rate', type=float, default=0.25, help='Sample sample_rate percent from initial edges.')
parser.add_argument('--ini_graph_percent', type=float, default=0.5, help='Train and test data percent.')
parser.add_argument('--anomaly_percent', type=float, default=0.05,
                    help='Anomaly injection with proportion of anomaly_percent.')
parser.add_argument('--snapshots_', type=int, default=1700, help='The snapshot size .')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50, help='Number of hidden units.')
parser.add_argument('--nmid1', type=int, default=70, help='Number of nmid1 units.')
parser.add_argument('--nmid2', type=int, default=100, help='Number of nmid2 units.')
parser.add_argument('--beta', type=float, default=3.0, help='Hyper-parameters in the score function.')
parser.add_argument('--mui', type=float, default=0.5, help='Hyper-parameters in the score function.')
parser.add_argument('--gama', type=float, default=0.6, help='Parameters in the score function.')
parser.add_argument('--w', type=int, default=3, help='Hyper-parameters in the score function.')
# parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()

data_path = '../munmun_digg_reply/out.munmun_digg_reply'
Net1 = GCN(nfeat=args.hidden, nmid1=args.nmid1, nmid2=args.nmid2, nhid=args.hidden, dropout=args.dropout)
Net2 = HCA(hidden=args.hidden, dropout=args.dropout)
Net3 = GRU(hidden=args.hidden, dropout=args.dropout)
Net4 = Score(beta=args.beta, mui=args.mui, hidden=args.hidden, dropout=args.dropout)
# snapshots_train, l_train, snapshots_test, l_test, nodes, n_train = snapshot(data_path=data_path, sample_rate=args.sample_rate,
#                                                                    ini_graph_percent=args.ini_graph_percent,
#                                                                    anomaly_percent=args.anomaly_percent,
#                                                                    snapshots_=args.snapshots_)
# np.savez("snapshot_25_5a_17_1.npz",snapshots_train = snapshots_train, l_train = l_train, snapshots_test = snapshots_test, l_test = l_test, nodes = nodes, n_train = n_train)
snapshots=np.load("snapshot_25_5a_17.npz", allow_pickle=True)
snapshots_test, l_test, nodes, n_train=snapshots['snapshots_test'], snapshots['l_test'], snapshots['nodes'], snapshots['n_train']
l_test = int(l_test)
nodes = int(nodes)
n_train = int(n_train)

print('===> Loding models...')
dir_ = './checkpoint/adj_.pth'
checkpoint_ = torch.load(dir_)
adj = checkpoint_['adj_']
# print('epoch{}'.format(0))
# dir = './checkpoint/NEWR_Adam_lr_0.001_w_3_epoch{}.pth'.format(16)
# dir = './checkpoint/S5_NEW_Adam_lr_0.001_w_3_epoch{}.pth'.format(39)
# dir = './checkpoint/Flag_Sparse_S03_NEW_Adam_lr_0.001_w_3_epoch{}.pth'.format(49)
dir = './checkpoint/NEW_Sparse_S17_Adam_lr_0.001_w_3_epoch{}.pth'.format(20)
checkpoint = torch.load(dir)
Net1.load_state_dict(checkpoint['Net1'])
Net2.load_state_dict(checkpoint['Net2'])
Net3.load_state_dict(checkpoint['Net3'])
Net4.load_state_dict(checkpoint['Net4'])
H_list = checkpoint['H_list']
nn.init.sparse_(H_list[-1][n_train: , :].t(), sparsity=0.9)
H_ = torch.zeros((args.w, nodes, args.hidden))
loss_a = checkpoint['loss_a']
print(loss_a)


def test():
    Net1.eval()
    Net2.eval()
    Net3.eval()
    Net4.eval()

    for i in range(l_test):
        global H_list, H_, adj
        data = snapshots_test[i]
        # snapshot = torch.from_numpy(data[:,(0,1)])
        snapshot_ = data[:, (0, 1)]
        label = data[:, 2]
        H = H_list[-1]
        prob = []
        for edge in snapshot_:
            m = edge[0]
            n = edge[1]
            score = Net4(hi=H[m - 1], hj=H[n - 1])
            prob.append(score.detach().numpy())
        prob = np.array(prob)
        auc = AUC(label=label, pre=prob)
        snapshot, acc = edge_p(edge=snapshot_, prob=prob, label=label, a=args.anomaly_percent)
        print('In snapshot {}'.format(i), 'the AUC results: {}.'.format(auc), 'the Acc results: {}.'.format(acc))
        snapshot = torch.from_numpy(snapshot)
        for j in range(args.w):
            H_[j] = H_list[-args.w + j]
        adj, Adj = update_adj(adj=adj, snapshot=snapshot, nodes=nodes)
        Adjn = normalize_adj(Adj + torch.eye(Adj.shape[0]))
        # H, H_, adjn, snapshot = Variable(H), Variable(H_), Variable(adjn), Variable(snapshot)
        current = Net1(x=H, adj=Adjn, Adj=Adj)
        short = Net2(C=H_)
        Hn = Net3(current=current, short=short)
        H_list = torch.cat([H_list, Hn.unsqueeze(0)], dim=0)


def edge_p(edge, prob, label, a):
    rank = np.argsort(prob)
    l = len(prob)
    normal_index = rank[:-int(np.floor(l * a))]
    anomaly_index = rank[-int(np.floor(l * a)):]
    edge_p = np.delete(edge, anomaly_index, axis=0)
    count = 0
    for i in (normal_index):
        if label[i] == 0:
            count = count + 1
    for j in (anomaly_index):
        if label[j] == 1:
            count = count + 1
    return edge_p, (count / l)
#
# def edge_p(edge, prob, label, a):
#     rank = np.argsort(prob)
#     l = len(prob)
#     normal_index = rank[:-int(np.floor(l * a))]
#     anomaly_index = rank[-int(np.floor(l * a)):]
#     neg_l = np.where(label == 1)
#     edge_p = np.delete(edge, neg_l, axis=0)
#     count = 0
#     for i in (normal_index):
#         if label[i] == 0:
#             count = count + 1
#     for j in (anomaly_index):
#         if label[j] == 1:
#             count = count + 1
#     return edge_p, (count / l)



if __name__ == "__main__":
    test()
