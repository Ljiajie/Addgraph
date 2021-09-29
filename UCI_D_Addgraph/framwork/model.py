# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: model.py
# @time: 2020/03/08
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import SpGraphAttentionLayer

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nheads*nhid,
                                             nout,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)
    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        flag = []
        D = adj.sum(1)
        for index, d in enumerate(D):
            if d.item() == 0:
                flag.append(index)
        x_ = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        h = self.out_att(x_, adj)
        h[flag] = x[flag]
        h = F.elu(h)
        return h

    def loss(self):
        loss = self.out_att.loss()
        for att in self.attentions:
            loss = loss + att.loss()
        return loss



class HCA(nn.Module):
    def __init__(self, hidden, dropout):
        super(HCA, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.Q = Parameter(torch.DoubleTensor(hidden, hidden), requires_grad=True)
        self.r = Parameter(torch.DoubleTensor(hidden), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt((self.Q.size(0)))
        self.Q.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.r.size(0))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, C):
        # C_=torch.einsum('wnh->nwh',C)
        # C_t=torch.einsum('nwh->nhw',C_)
        C_ = C.permute(1, 0, 2)
        C_t = C_.permute(0, 2, 1)
        e_ = torch.einsum('ih,nhw->niw', self.Q, C_t)
        e_ = F.dropout(e_, self.dropout, training=self.training)
        e = torch.einsum('h,nhw->nw', self.r, torch.tanh(e_))
        e = F.dropout(e, self.dropout, training=self.training)
        a = F.softmax(e, dim=1)
        short = torch.einsum('nw,nwh->nh', a, C_)
        # short_t = torch.transpose(short, 0, 1)
        return short

    def loss(self):
        loss = torch.norm(self.Q, 2).pow(2) + torch.norm(self.r, 2).pow(2)
        return loss


class GRU(nn.Module):
    def __init__(self, hidden, dropout):
        super(GRU, self).__init__()
        self.dropout = dropout
        self.Up = Parameter(torch.DoubleTensor(hidden, hidden))
        self.Wp = Parameter(torch.DoubleTensor(hidden, hidden))
        self.bp = Parameter(torch.DoubleTensor(hidden))
        self.Ur = Parameter(torch.DoubleTensor(hidden, hidden))
        self.Wr = Parameter(torch.DoubleTensor(hidden, hidden))
        self.br = Parameter(torch.DoubleTensor(hidden))
        self.Uc = Parameter(torch.DoubleTensor(hidden, hidden))
        self.Wc = Parameter(torch.DoubleTensor(hidden, hidden))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Up.size(0))
        self.Up.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Wp.size(0))
        self.Wp.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.bp.size(0))
        self.bp.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Ur.size(0))
        self.Ur.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Wr.size(0))
        self.Wr.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.br.size(0))
        self.br.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Uc.size(0))
        self.Uc.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.Wc.size(0))
        self.Wc.data.uniform_(-stdv, stdv)

    def forward(self, current, short):
        # 更新门
        P = torch.sigmoid(torch.matmul(current, self.Up) + torch.matmul(short, self.Wp) + self.bp)
        P = F.dropout(P, self.dropout, training=self.training)
        # 重置门
        R = torch.sigmoid(torch.matmul(current, self.Ur) + torch.matmul(short, self.Wr) + self.br)
        R = F.dropout(R, self.dropout, training=self.training)
        # 候选隐藏状态
        H_tilda = torch.tanh(torch.matmul(current, self.Uc) + R * torch.matmul(short, self.Wc))
        H_tilda = F.dropout(H_tilda, self.dropout, training=self.training)
        # 隐藏状态
        H = (1 - P) * short + P * H_tilda
        return H

    def loss(self):
        loss1 = torch.norm(self.Up, 2).pow(2) + torch.norm(self.Wp, 2).pow(2) + torch.norm(self.bp, 2).pow(2)
        loss2 = torch.norm(self.Ur, 2).pow(2) + torch.norm(self.Wr, 2).pow(2) + torch.norm(self.br, 2).pow(2)
        loss3 = torch.norm(self.Uc, 2).pow(2) + torch.norm(self.Wc, 2).pow(2)
        return (loss1 + loss2 + loss3)


class Score(nn.Module):
    def __init__(self, beta, mui, hidden, dropout):
        super(Score, self).__init__()
        self.a = Parameter(torch.DoubleTensor(hidden), requires_grad=True)
        self.b = Parameter(torch.DoubleTensor(hidden), requires_grad=True)
        self.beta = beta
        self.mui = mui
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.b.size(0))
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, hi, hj):
        s = self.a * hi + self.b * hj
        s = F.dropout(s, self.dropout, training=self.training)
        s_ = torch.norm(s, 2).pow(2)
        # score=torch.sigmoid(self.beta * s_ - self.mui)
        x = self.beta * s_ - self.mui
        score = 1.0 / (1 + torch.exp(-x))
        return score

    def loss(self):
        loss = torch.norm(self.a, 2).pow(2) + torch.norm(self.b, 2).pow(2)
        return loss


if __name__ == "__main__":
    # Net=GCN(nfeat=3, nmid=5, nhid=3)
    # x=torch.rand(6,3)
    # adj=torch.rand(6,6)
    # h=Net(x,adj)
    # loss=Net.loss_c()
    # print(x.type())
    # Net1 = GCN(nfeat=100, nmid1=200, nmid2=150, nhid=100)
    # H = torch.zeros(981, 100)
    # adj_ = torch.FloatTensor(981, 981)
    # current = Net1(x=H, adj=adj_)
    # print(Net1.gc1.weight)
    # current=torch.FloatTensor([[1,2],[3,4],[4,5],[6,7]])
    # short=torch.FloatTensor([[3,6],[1,2],[6,2],[4,2]])
    # net=GRU(2)
    # h=net(current,short)
    # loss=net.loss() #调用函数即使没有参数也得加括号
    # print(h)
    # print('\n',loss)
    # model=net
    # for k, v in model.named_parameters():
    #     print(k, v.size())
    net = SpGAT(nfeat=11, nhid=15, nout=11, dropout=0.2, alpha=0.2, nheads=8)
    net2= Score(beta=0.2, mui=0.6, hidden=11, dropout=0.2)
    h = torch.rand(8,11)
    adj = torch.tensor([[0,1,1,0,0,0,1,0],[0,0,0,0,0,0,0,0],[0,0,0,1,0,2,0,0],[0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,0],[1,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
    h_ = net(h, adj)
    loss = net.loss()
    loss2 = net2(h_[1], h_[2])
    print(loss2)
    loss2.backward()