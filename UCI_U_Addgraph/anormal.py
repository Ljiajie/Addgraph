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
a1=[0.8863556249777665,0.8746331377004427,0.8237180952380952,0.8313499619192688,0.8043661966766275,0.8468934240362812]
a5=[0.8615261315843652,0.869959010974481,0.8256744071146245,0.813014359498269,0.8037808071256254,0.7949842176598474]
a10=[0.8591363063223801,0.8383239400019596,0.8173613917109213,0.8213630136320796,0.7997132973626849,0.8133052956801169]
a15=[0.8592805881423385,0.8240598485629947,0.7998277421928627,0.8094073094689223,0.7921544348028371,0.780208390569693]
a20 = [0.8346924155222611,0.8237495332371012,0.7880275215543943,0.7764430566169511,0.7855067087252591,0.759804275920147]
u=[0.8152788602023686,0.8253189583221608,0.8281564889928686,0.8266893855220513,0.8145806452780658]
d=[0.8234780744181124,0.8515145148361101,0.8541033965381777,0.8506022941913071,0.8408061573574112]
U_Model=[]
# Snapshot=np.arange(5)+1
Snapshot = [1,2,3,4,5]
plt.xlabel('w')  # 设置x，y轴的标签
plt.ylabel('AUC')
plt.xticks(Snapshot)
# plt.plot(Snapshot, a1, 'g^-', label='anomaly_percent:1%')
# plt.plot(Snapshot, a5, 'ro-', label='anomaly_percent:5%')
# plt.plot(Snapshot, a10, 'bD-', label='anomaly_percent:10%')
# plt.plot(Snapshot, a15, color='#900302', marker='s', label='anomaly_percent:15%')
# plt.plot(Snapshot, a20, 'k*-', label='anomaly_percent:20%')
# plt.plot(Snapshot, DeepWalk, 'g^-', label='DeepWalk')
plt.plot(Snapshot, u, 'bD-', label='U_Model')
plt.plot(Snapshot, d, 'ro-', label='D_Model')
# plt.legend()
plt.legend(loc="upper right")
plt.show()