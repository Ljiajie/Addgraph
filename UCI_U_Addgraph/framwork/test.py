# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: test.py
# @time: 2020/03/09
# import numpy as np
# edges = np.loadtxt('../opsahl-ucsocial/out.opsahl-ucsocial', dtype=int, comments='%',usecols=(0, 1))
# idx_reverse = np.nonzero(edges[:, 0] - edges[:, 1] > 0)
# tmp = edges[idx_reverse]
# tmp[:, [0, 1]] = tmp[:, [1, 0]]
# edges[idx_reverse] = tmp
# idx_remove_dups = np.nonzero(edges[:, 0] - edges[:, 1] < 0)
# edges = edges[idx_remove_dups]
# edges = edges[:, 0:2]
# # edges = edges[:, 0:2]
# print(edges.shape)
# print(edges[0:10,:])
import torch
import math
# a=torch.tensor([1.0,2,3])
# b=a.sqrt()
# print(math.sqrt((b.size(0))))
# print(a.type())
# a=[1,2]
# print(len(a))
# H_=torch.zeros((2,3))
# H_=H_.float()
# print(H_.type())
def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    D = adj.sum(1)
    r_inv_sqrt = D.pow(-0.5)
    # r_inv_sqrt[torch.eq(r_inv_sqrt, float('inf'))] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(adj, r_mat_inv_sqrt).t(), r_mat_inv_sqrt)
adj=torch.FloatTensor([[1,1,1,0],[1,1,1,0],[1,1,1,1],[0,0,1,1]])
a_=normalize_adj(adj)
print(adj)
print(a_)