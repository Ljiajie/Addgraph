# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: anormal.py
# @time: 2020/05/22
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
a1=[0.9050781790507818,0.8708851227222323,0.8480532200175857,0.8360767471857411,0.8706650152439024,0.9176484419473802]
a5=[0.8674311871001761,0.8522179143712424,0.8344530391996844,0.8295769220428904,0.8493126856282707,0.891628630886802]
a10=[0.8709060907625096,0.8733782834036147,0.8304294592683562,0.829808614782932,0.8746821703886707,0.8601199574304489]
a15=[0.8585651265650446,0.8565099260751434,0.8134486201468129,0.8175329131476937,0.8710405385468633,0.8487976836141073]
a20 = [0.8640836127700946,0.8478662642915734,0.8291931608275749,0.8259659977344973,0.8579856391207963,0.8394472222635588]
SC = [0.6125,0.623,0.62,0.625,0.628,0.632]
U_Model=[]
Snapshot=np.arange(6)+1
plt.xlabel('Snapshot')  # 设置x，y轴的标签
plt.ylabel('AUC')
plt.plot(Snapshot, a1, 'g^-', label='anomaly_percent:1%')
plt.plot(Snapshot, a5, 'ro-', label='anomaly_percent:5%')
plt.plot(Snapshot, a10, 'bD-', label='anomaly_percent:10%')
plt.plot(Snapshot, a15, color='#900302', marker='s', label='anomaly_percent:15%')
plt.plot(Snapshot, a20, 'k*-', label='anomaly_percent:20%')

# plt.plot(Snapshot, SC, 'cv-', label='SC')
# plt.legend()
plt.legend(loc="upper right")
plt.show()