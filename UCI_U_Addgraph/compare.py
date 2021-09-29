# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: compare.py
# @time: 2020/05/11
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = Axes3D(fig)

x = np.array([25,50,100,200,400])
X = np.log10(x)
Y = np.arange(5) + 1
X, Y = np.meshgrid(X, Y)

# Z = np.array(
#     [
#     [0.7849432853778776, 0.7900000000000000, 0.8000000000000000,0.8081087936382515, 0.7985773524366172],
#     [0.7860938312374598, 0.7902866118758104, 0.8000000000000000,0.8065670206501329, 0.8147494937971081],
#     [0.7803057564911199, 0.7902002087776203, 0.8000000000000000,0.8115627371849246, 0.8249242448467674],
#     [0.7861007432552363, 0.7958772348242385, 0.8000000000000000,0.7851775701318472, 0.8107105653729302],
#     [0.7846900726830554, 0.7959230888914906, 0.8000000000000000,0.8154699787505825, 0.8040406085077971]
# ]
# )
Z = np.array(
    [
    [0.7949432853778776, 0.8070636536254524, 0.8183040989481585,0.8081087936382515, 0.7985773524366172],
    [0.8096400317039839, 0.8002866118758104, 0.8207100325485294,0.8065670206501329, 0.8147494937971081],
    [0.8103057564911199, 0.8202002087776203, 0.8323093477117394,0.8115627371849246, 0.8249242448467674],
    [0.8061007432552363, 0.8058772348242385, 0.8051116518987044,0.7851775701318472, 0.8107105653729302],
    [0.7754308683809645, 0.8259230888914906, 0.8246900726830554,0.8154699787505825, 0.8040406085077971]
]
)
Z = Z.T

# 绘制三维曲面图
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#设置三个坐标轴信息
ax.set_xlabel('Log(d)', color='b')
ax.set_ylabel('Layers', color='g')
ax.set_zlabel('AUC', color='r')
ax.set_zlim(0.78, 0.85)
# ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# plt.colorbar()
# plt.draw()
fig.colorbar(surf, shrink=0.6, aspect=5)
plt.show()

# plt.savefig('3D.jpg')