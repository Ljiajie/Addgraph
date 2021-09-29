# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: negative_sample.py
# @time: 2020/03/11
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn


# def update_adj(adj,snapshot):
#     data = tuple(map(tuple, snapshot))
#     for i, j in data:
#         adj[i - 1][j - 1] = adj[i - 1][j - 1] + 1  # Convert to 0-based index.
#         adj[i - 1][j - 1] = adj[i - 1][j - 1] + 1  # Convert to 0-based index.
#     return adj
def update_adj(adj, snapshot, nodes):
    Adj = torch.zeros((nodes, nodes))
    for edge in snapshot:
        adj[edge[0].item() - 1][edge[1].item() - 1] = adj[edge[0].item() - 1][
                                                          edge[1].item() - 1] + 1  # Convert to 0-based index.
        adj[edge[1].item() - 1][edge[0].item() - 1] = adj[edge[1].item() - 1][
                                                          edge[0].item() - 1] + 1  # Convert to 0-based index.
        Adj[edge[0].item() - 1][edge[1].item() - 1] = Adj[edge[0].item() - 1][edge[1].item() - 1] + 1
        Adj[edge[1].item() - 1][edge[0].item() - 1] = Adj[edge[1].item() - 1][edge[0].item() - 1] + 1
    return adj, Adj


class negative_sample(nn.Module):
    def __init__(self):
        super(negative_sample, self).__init__()

    def forward(self, adj, Adj, snapshot, H, f, arg):
        if arg:
            data = tuple(map(tuple, snapshot.data.cpu().numpy()))
            D = adj.sum(1)
            D_ = Adj.sum(1)
            D = D.data.cpu().numpy()
            D_ = D_.data.cpu().numpy()
            adj = adj.data.cpu().numpy()
        else:
            data = tuple(map(tuple, np.array(snapshot)))
            D = adj.sum(1)
            D_ = Adj.sum(1)
            D = np.array(D)
            D_ = np.array(D_)
            adj = np.array(adj)
        # data = tuple(map(tuple, snapshot.data.cpu().numpy()))
        len = snapshot.shape[0]
        # for i,j in data:
        #     adj[i - 1][j - 1] = adj[i - 1][j - 1]+1  # Convert to 0-based index.
        #     adj[i - 1][j - 1] = adj[i - 1][j - 1]+1  # Convert to 0-based index.
        # D = adj.sum(1)
        # D = D.data.cpu().numpy()
        # adj = adj.data.cpu().numpy()
        th = (np.argwhere(D == 0) + 1)[0].item()
        # n_data=[]
        n_loss = torch.zeros(len)
        index = 0
        print('----------Begin----------')
        for i, j in tqdm(data):
            di = D_[i - 1]
            dj = D_[j - 1]
            pi = float(di) / (di + dj)
            pj = float(dj) / (di + dj)
            d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj])
            d = d.item()
            d_ = np.argwhere(adj[d - 1] == 0) + 1
            d_ = (np.squeeze(d_))
            id1 = d_ < th
            d_ = d_[id1]
            id2 = d_ != d
            d_ = d_[id2]
            dn = np.random.choice(a=d_, size=1, replace=False)
            dn = dn.item()
            if d > dn:
                a = dn
                b = d
            else:
                a = d
                b = dn
            loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[a - 1], hj=H[b - 1])
            count = 0
            while (loss > 0):  #
                # f function
                count=count+1
                d = np.random.choice(a=[j, i], size=1, replace=False, p=[pi, pj])
                d = d.item()
                d_ = np.argwhere(adj[d - 1] == 0) + 1
                d_ = (np.squeeze(d_))
                id1 = d_ < th
                d_ = d_[id1]
                id2 = d_ != d
                d_ = d_[id2]
                dn = np.random.choice(a=d_, size=1, replace=False)
                dn = dn.item()
                if d > dn:
                    a = dn
                    b = d
                else:
                    a = d
                    b = dn
                loss = f(hi=H[i - 1], hj=H[j - 1]) - f(hi=H[a - 1], hj=H[b - 1])
                if (count>1899):
                    loss = torch.zeros(1)
                    break
                # # if d > dn:
                #     t = d
                #     d = dn
                #     dn = t
            # n_data.append([a, b])
            n_loss[index] = loss
            index = index + 1
        print('----------End----------')
        # return np.array(n_data),n_loss
        return n_loss
