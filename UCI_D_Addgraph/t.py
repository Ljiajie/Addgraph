# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: t.py
# @time: 2020/04/02
import numpy as np
import torch
# snapshots=np.load("snapshot_10a.npz", allow_pickle=True)
# snapshots_train, l_train, nodes=snapshots['snapshots_train'], snapshots['l_train'], snapshots['nodes']
# l_train = int(l_train)
# nodes = int(nodes)
# print(type(l_train),type(nodes))

# dir_ = './checkpoint/adj.pth'
# checkpoint_ = torch.load(dir_)
# ADJ = checkpoint_['adj']
# print(ADJ[451-1][1191-1])

import matplotlib.pyplot as plt
x = np.arange(-8,8,0.2)
y1 = 1.0 /(1.0 + np.exp(-x))
y2 = 1.0 /(1.0 + np.exp(x))
# plt.plot(x,y1)
plt.plot(x,y2)
plt.show()
