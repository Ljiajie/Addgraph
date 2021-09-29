# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: tt.py
# @time: 2020/04/18
import torch
import torch.nn as nn
dir = './checkpoint/Flag_Sparse_S03_NEW_Adam_lr_0.001_w_3_epoch{}.pth'.format(18)
checkpoint = torch.load(dir)
# Net1.load_state_dict(checkpoint['Net1'])
# Net2.load_state_dict(checkpoint['Net2'])
# Net3.load_state_dict(checkpoint['Net3'])
# Net4.load_state_dict(checkpoint['Net4'])
H_list = checkpoint['H_list']
for i in range(len(H_list)):
    print(H_list[i])