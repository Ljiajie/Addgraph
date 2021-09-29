# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: layers.py
# @time: 2020/03/08
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.autograd import Variable

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b



class SpecialSpmm(Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
        # a = torch.sparse_coo_tensor(indices, values, shape)
        # return torch.spmm(a,b)


class SpGraphAttentionLayer(Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        # nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.W = Parameter(torch.DoubleTensor(in_features, out_features))
        # self.a = Parameter(torch.zeros(size=(1, 2 * out_features)), requires_grad=True)
        # nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a = Parameter(torch.DoubleTensor(1, 2*out_features))
        self.reset_parameters()
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()


    def reset_parameters(self):
        stdv = 1. / math.sqrt((self.W.size(1)))
        self.W.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()
        # print(self.W)
        # print(input)
        h = torch.mm(input, self.W)
        h = F.dropout(h, self.dropout, training=self.training)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e = F.dropout(edge_e, self.dropout, training=self.training)
        #
        # a_ = self.a.mm(edge_h)
        # a_ = F.dropout(a_, self.dropout, training=self.training)
        # edge_e = torch.exp(-self.leakyrelu(a_).squeeze())
        # assert not torch.isnan(edge_e).any()


        # edge_e: E

        #consider weight
        adj_w = torch.zeros(edge_e.shape[0])
        for i in range(edge_e.shape[0]):
            adj_w[i] = adj[edge[0][i]][edge[1][i]]
        edge_e = edge_e * adj_w
        # print(edge_e)
            # edge_e[i].item() =  edge_e[i].item() *adj[edge[0][i]][edge[1][i]]
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # a = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N])).to_dense()
        # e_rowsum = torch.matmul(a, torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        flag = []
        for i, e_row in enumerate(e_rowsum):
            if e_row.item() == 0:
                e_rowsum[i] = torch.ones(1)
                flag.append(i)


        # edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # a_ = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N])).to_dense()
        # h_prime = torch.matmul(a_, h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        # h_prime[flag] = input[flag]
        h_prime[flag] = h[flag]

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def loss(self):
        loss = torch.norm(self.W, 2).pow(2) + torch.norm(self.a, 2).pow(2)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
