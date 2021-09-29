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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--sample_rate', type=float, default=1, help='Sample sample_rate percent from initial edges.')
parser.add_argument('--ini_graph_percent', type=float, default=0.5, help='Train and test data percent.')
parser.add_argument('--anomaly_percent', type=float, default=0.00004,
                    help='Anomaly injection with proportion of anomaly_percent.')
parser.add_argument('--snapshots_', type=int, default=5300, help='The snapshot size .')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--nmid1', type=int, default=100, help='Number of nmid1 units.')
parser.add_argument('--nmid2', type=int, default=100, help='Number of nmid2 units.')
# parser.add_argument('--nmid3', type=int, default=25, help='Number of nmid2 units.')
# parser.add_argument('--nmid4', type=int, default=400, help='Number of nmid2 units.')
parser.add_argument('--beta', type=float, default=1.0, help='Hyper-parameters in the score function.')
parser.add_argument('--mui', type=float, default=0.3, help='Hyper-parameters in the score function.')
parser.add_argument('--gama', type=float, default=0.6, help='Parameters in the score function.')
parser.add_argument('--w', type=int, default=5, help='Hyper-parameters in the score function.')
# parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()

data_path = 'opsahl-ucsocial/out.opsahl-ucsocial'
# Net1 = GCN(nfeat=args.hidden, nmid1=args.nmid1, nmid2=args.nmid2, nhid=args.hidden, dropout=args.dropout)
Net1 = GCN(nfeat=args.hidden, nmid1=args.nmid1, nmid2=args.nmid2, nhid=args.hidden, dropout=args.dropout)
Net2 = HCA(hidden=args.hidden, dropout=args.dropout)
Net3 = GRU(hidden=args.hidden, dropout=args.dropout)
Net4 = Score(beta=args.beta, mui=args.mui, hidden=args.hidden, dropout=args.dropout)
snapshots_train, l_train, snapshots_test, l_test, nodes, n_train = snapshot(data_path=data_path, sample_rate=args.sample_rate,
                                                                   ini_graph_percent=args.ini_graph_percent,
                                                                   anomaly_percent=args.anomaly_percent,
                                                                   snapshots_=args.snapshots_)
np.savez("snapshot_0a53.npz", snapshots_train=snapshots_train, l_train=l_train, snapshots_test=snapshots_test,
         l_test=l_test, nodes=nodes, n_train=n_train)
print('===> Loding data...')
snapshots=np.load("snapshot_0a53.npz", allow_pickle=True)
# snapshots=np.load("a10_i50_snapshot_10a53.npz", allow_pickle=True)
snapshots_test, l_test, nodes, n_train=snapshots['snapshots_test'], snapshots['l_test'], snapshots['nodes'], snapshots['n_train']
l_test = int(l_test)
nodes = int(nodes)
n_train = int(n_train)

print('===> Loding models...')
dir_ = './checkpoint/adj.pth'
checkpoint_ = torch.load(dir_)
adj = checkpoint_['adj']
# print('epoch{}'.format(0))
# dir = './checkpoint/NEWR_Adam_lr_0.001_w_3_epoch{}.pth'.format(16)
# dir = './checkpoint/S5_NEW_Adam_lr_0.001_w_3_epoch{}.pth'.format(30)
# dir = './checkpoint/Flag_S53_Adam_lr_0.001_w_3_epoch{}.pth'.format(37)
# dir = './checkpoint/SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(29)
# dir = './checkpoint/mid75150_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(59)
# dir = './checkpoint/mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(49)
# dir = './checkpoint/mid2300_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(39)
# dir = './checkpoint/mid2300_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(epoch)
# dir = './checkpoint/n_train_mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(31)
# dir = './checkpoint/headtail_mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(59)
# dir = './checkpoint/mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(30)
# dir = './checkpoint/l3_mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(28)#28
# dir = './checkpoint/a20_mid100_SPARE_S60_lr_0.001_w_3_epoch{}.pth'.format(39)
dir = './checkpoint/mid100_SPARE_S53_lr_0.001_w_5_epoch{}.pth'.format(26)
checkpoint = torch.load(dir)
Net1.load_state_dict(checkpoint['Net1'])
Net2.load_state_dict(checkpoint['Net2'])
Net3.load_state_dict(checkpoint['Net3'])
Net4.load_state_dict(checkpoint['Net4'])
H_list = checkpoint['H_list']
H_ = torch.zeros((args.w, nodes, args.hidden))
loss_a = checkpoint['loss_a']
print(loss_a)
nn.init.sparse_(H_list[-1][n_train: , :], sparsity=0.9)
# stdv = 1. / math.sqrt(H_list[-1].size(1))
# H_list[-1][n_train:, :].data.uniform_(-stdv, stdv)

def test():
    Net1.eval()
    Net2.eval()
    Net3.eval()
    Net4.eval()
    # writer1 = SummaryWriter('auc')
    Netwalk=[0.77,0.785,0.75,0.74,0.7125,0.725]
    Goutlier=[0.70,0.7125,0.695,0.67,0.687,0.695]
    CM_sketch=[0.71,0.719,0.68,0.679,0.68,0.675]
    DeepWalk = [0.735,0.755,0.711,0.722,0.705,0.708]
    SC = [0.6125,0.623,0.62,0.625,0.628,0.632]
    U_Model=[]
    for i in range(l_test):
        global H_list, H_, adj, n_train
        data = snapshots_test[i]
        # snapshot = torch.from_numpy(data[:,(0,1)])
        snapshot_ = data[:, (0, 1)]
        label = data[:, 2]
        # n_test_ = np.unique(snapshot_)
        # n_test = len(n_test_)
        # nn.init.sparse_(H_list[-1][n_train:n_test, :], sparsity=0.9)
        # stdv = 1. / math.sqrt(H_list[-1].size(1))
        # H_list[-1][n_train:n_test, :].data.uniform_(-stdv, stdv)
        # n_train = n_test
        H = H_list[-1]
        prob = []
        for edge in snapshot_:
            m = edge[0]
            n = edge[1]
            score = Net4(hi=H[m - 1], hj=H[n - 1])
            prob.append(score.detach().numpy())
        prob = np.array(prob)
        auc = AUC(label=label, pre=prob)
        U_Model.append(auc)
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
        # writer1.add_scalar('U_AUC', auc, i+1)
    sum = 0
    for i in U_Model:
        sum =sum + i
    print(sum/len(U_Model))
    Snapshot=np.arange(l_test)+1
    plt.xlabel('Snapshot')  # 设置x，y轴的标签
    plt.ylabel('AUC')
    plt.plot(Snapshot, U_Model, 'ro-', label='U_Model')
    plt.plot(Snapshot, Netwalk, 'bD-', label='Netwalk')
    plt.plot(Snapshot, Goutlier, color='#900302', marker='s', label='Goutlier')
    plt.plot(Snapshot, CM_sketch, 'k*-', label='CM_sketch')
    plt.plot(Snapshot, DeepWalk, 'g^-', label='DeepWalk')
    plt.plot(Snapshot, SC, 'cv-', label='SC')
    plt.legend()
    plt.show()
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
