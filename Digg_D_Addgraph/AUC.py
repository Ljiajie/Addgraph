# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: AUC.py
# @time: 2020/03/19
def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 0]
    neg = [i for i in range(len(label)) if label[i] == 1]

    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] < pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return auc / (len(pos) * len(neg))
