# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: create_adj.py
# @time: 2020/04/22
import torch
import numpy as np
from framwork.negative_sample import *
# snapshots=np.load("snapshot_25_5a_17.npz", allow_pickle=True)
# snapshots_train, l_train, nodes, n_train=snapshots['snapshots_train'], snapshots['l_train'], snapshots['nodes'], snapshots['n_train']
# l_train = int(l_train)
# nodes = int(nodes)
# n_train = int(n_train)
# adj = torch.zeros((nodes, nodes))
# for i in range(l_train):
#     snapshot = torch.from_numpy(snapshots_train[i])
#     adj, Adj = update_adj(adj=adj, snapshot=snapshot, nodes=nodes)
# adj = {'adj_': adj}
# dir = './checkpoint/adj_.pth'
# torch.save(adj, dir)
dir_ = './checkpoint/adj_.pth'
checkpoint_ = torch.load(dir_)
adj = checkpoint_['adj_']
print(adj[10 - 1][7124  - 1])
print(adj[5521 - 1][7507  - 1])
print(adj[4270 - 1][6852  - 1])
print(adj[785 - 1][5548  - 1])
print(adj[365 - 1][2207  - 1])
print(adj[2207 - 1][365  - 1])