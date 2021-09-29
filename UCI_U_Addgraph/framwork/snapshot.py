# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: snapshot.py
# @time: 2020/03/10
import numpy as np
from anomaly_generation import anomaly_generation
from load_uci_messages import load_uci_messages
from negative_sample import negative_sample
from model import *
from tqdm import tqdm


def snapshot(data_path, sample_rate, ini_graph_percent, anomaly_percent, snapshots_):
    data, n, m = load_uci_messages(data_path, sample_rate)
    n_train, train, synthetic_test = anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m)
    # train_snaps = int(np.ceil(np.size(train, 0) / snapshots_))
    l_train = np.size(train, 0)
    snapshots_train = []
    current = 0
    while (l_train - current) >= snapshots_:
        snapshots_train.append(train[current:current + snapshots_])
        current += snapshots_
    snapshots_train.append(train[current:])
    print("Train data:number of snapshots: %d, edges in each snapshot: %d" % (len(snapshots_train), snapshots_))

    l_test = np.size(synthetic_test, 0)
    snapshots_test = []
    current = 0
    while (l_test - current) >= snapshots_:
        snapshots_test.append(synthetic_test[current:current + snapshots_])
        current += snapshots_
    snapshots_test.append(synthetic_test[current:])
    print("Test data:number of snapshots: %d, edges in each snapshot: %d" % (len(snapshots_test), snapshots_))
    return snapshots_train, len(snapshots_train), snapshots_test, len(snapshots_test), n, n_train


# def normalize_adj(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
#     r_mat_inv_sqrt = np.diag(r_inv_sqrt)
#     return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    D = adj.sum(1)
    r_inv_sqrt = D.pow(-0.5)
    r_inv_sqrt[torch.eq(r_inv_sqrt, float('inf'))] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(adj, r_mat_inv_sqrt).t(), r_mat_inv_sqrt)


if __name__ == "__main__":
    nfeat = nhid = hidden = 100
    nmid1 = 200
    nmid2 = 150

    Net1 = GCN(nfeat=nfeat, nmid1=nmid1, nmid2=nmid2, nhid=nhid, dropout=0)
    Net2 = HCA(hidden=hidden, dropout=0)
    Net3 = GRU(hidden=hidden, dropout=0)
    Net4 = Score(beta=1.0, mui=0.3, hidden=hidden, dropout=0)

    data_path = '../opsahl-ucsocial/out.opsahl-ucsocial'
    torch.set_default_tensor_type(torch.FloatTensor)
    snapshots_train, l_train, snapshots_test, l_test, nodes = snapshot(data_path, 0.3, 0.5, 0.2, 1000)
    H = torch.rand(nodes, hidden)
    H_ = torch.zeros(3, nodes, hidden)
    adj = np.zeros((nodes, nodes))
    snapshot = snapshots_train[0]
    n_data, n_loss, adj = negative_sample(adj, snapshots_train[0], H, Net4)
    adjn = normalize_adj(adj + np.eye(adj.shape[0]))
    adj_ = torch.from_numpy(adjn)
    # print(n_data)
    print(n_loss)
    # print(n_loss.shape)
    current = Net1(x=H, adj=adj_)
    short = Net2(C=H_)
    Hn = Net3(current=current, short=short)
    # for i in tqdm(range(len(n_data))):
    #     normal=snapshot[i]
    #     anormal=n_data[i]
    #     score1=Net4(hi=Hn[normal[0]-1] ,hj=Hn[normal[1]-1])
    #     score2 = Net4(hi=Hn[anormal[0] - 1], hj=Hn[anormal[1] - 1])
    #     print(i+1)
    #     print(score1,',',score2)
    #     print('\n')

    loss = Net1.loss() + Net2.loss() + Net3.loss()
    print(loss[0])
    # print(Hn.shape)
    print(loss.grad)
    # n1_data, adj = negative_sample(adj, snapshots_train[1])
    # print(n1_data,adj)
    # print('\n')
    # print(n1_data.shape, adj.shape)
    # print(snapshots_test[10])
    # print(np.array(snapshots_test[0][:,0:2]))
