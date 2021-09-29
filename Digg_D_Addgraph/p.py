# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: p.py
# @time: 2020/05/15
import numpy as np
import matplotlib.pyplot as plt
Netwalk = [0.72, 0.77, 0.765, 0.778, 0.79, 0.798, 0.783]
Goutlier = [0.675, 0.725, 0.705, 0.72, 0.747, 0.745,0.74]
CM_sketch = [0.655, 0.678, 0.68, 0.71, 0.722, 0.724,0.71]
DeepWalk = [0.682, 0.735, 0.730, 0.732, 0.745, 0.75,0.745]
SC = [0.58, 0.598, 0.59, 0.602, 0.625, 0.64, 0.6]
D_Model = [0.8241191956533915, 0.809657167696375, 0.8160390780612728, 0.8286492944658321, 0.8306249463138085, 0.8057568814254274, 0.7421675490501402]
U_Model = [0.879425641025641, 0.8796620190331657, 0.8698423914761769,  0.819887668379063, 0.8159671710962589, 0.7969366877035244, 0.7879155670478679]
Snapshot=np.arange(7)+1
plt.xlabel('Snapshot')  # 设置x，y轴的标签
plt.ylabel('AUC')
plt.plot(Snapshot, U_Model, 'ro-', label='U_Model')
plt.plot(Snapshot, Netwalk, 'bD-', label='Netwalk')
plt.plot(Snapshot, Goutlier, color='#900302', marker='s', label='Goutlier')
plt.plot(Snapshot, CM_sketch, 'k*-', label='CM_sketch')
plt.plot(Snapshot, DeepWalk, 'g^-', label='DeepWalk')
plt.plot(Snapshot, SC, 'cv-', label='SC')
# plt.legend()
plt.legend(loc="upper right")
plt.show()