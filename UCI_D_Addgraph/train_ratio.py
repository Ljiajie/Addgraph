# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: train_ratio.py
# @time: 2020/05/13
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(8,5), dpi=120)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
x = (np.array([1908.3, 3158.2, 4140.6, 5510.2, 2015.3, 3235.0, 4453.8, 5798.4,
               2147.6, 3385.8, 4731.2, 5925.6, 2222.5, 3447.2, 5046.1, 6254.4]),
     np.array([9548.0, 11127.5, 11887.0, 13102.3, 10641.7, 12312.9, 12790.3,
               13915.8, 11320.0, 13300.1, 14024.3, 15461.0, 13146.6, 15219.9]),
     np.array([9873.6, 9757.7, 9684.9, 10581.7, 11429.4, 11178.6, 11089.3, 12002.6,
               12827.3, 12508.9, 12501.8, 13583.8, 14456.4, 13870.2, 13946.9])
     )
labels = ["第一产业", "第二产业", "第三产业"]
plt.boxplot(x, notch=False, labels=labels, meanline=True, showmeans=True)
plt.title("生产总值箱线图")
plt.xlabel("产业", verticalalignment="top")
plt.ylabel("生产总值（亿元）", rotation=0, horizontalalignment="right")
plt.savefig("./生产总值箱线图.png")
plt.show()