# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: view_H.py
# @time: 2020/04/21
import torch
dir = './checkpoint/S53_Adam_lr_0.001_w_3_epoch{}.pth'.format(22)
checkpoint = torch.load(dir)
H_list = checkpoint['H_list']
for i in range(len(H_list)):
    print(H_list[i])