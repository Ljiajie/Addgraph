# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: test2.py
# @time: 2020/03/15
import torch
from snapshot import *


def update_adj(adj, snapshot):
    for edge in snapshot:
        adj[edge[0].item() - 1][edge[1].item() - 1] = adj[edge[0].item() - 1][
                                                          edge[1].item() - 1] + 1  # Convert to 0-based index.
        adj[edge[1].item() - 1][edge[0].item() - 1] = adj[edge[1].item() - 1][
                                                          edge[0].item() - 1] + 1  # Convert to 0-based index.
    return adj


def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    D = adj.sum(1)
    r_inv_sqrt = D.pow(-0.5)
    r_inv_sqrt[torch.eq(r_inv_sqrt, float('inf'))] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(adj, r_mat_inv_sqrt).t(), r_mat_inv_sqrt)


# data_path = '../opsahl-ucsocial/out.opsahl-ucsocial'
# torch.set_default_tensor_type(torch.FloatTensor)
# snapshots_train, l_train, snapshots_test, l_test, nodes = snapshot(data_path, 1, 0.5, 0.05, 1000)
# adj=torch.zeros(nodes,nodes)
# for i in range(l_train):
#     snapshot = torch.from_numpy(snapshots_train[i])
#     adj = update_adj(adj, snapshot)
#     print(snapshot[1][1])
import numpy as np
fake_edges=np.array([[2,1],[6,9],[7,5],[4,3],[6,8],[5,2],[1,8],[2,6]])
anomaly_num=4
index=[]
data=np.array([[2,1],[6,8],[4,5],[4,3],[6,2],[5,2]])
for i in range(np.size(fake_edges, 0)):
    if len(index) < anomaly_num:
        for edge in data:
            if fake_edges[i][0] == edge[0] and fake_edges[i][1] == edge[1] :
                break
            else :
                continue
            index.append(i)