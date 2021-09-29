# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: loss_a_view.py
# @time: 2020/04/01
import torch
from tensorboardX import SummaryWriter

writer1 = SummaryWriter('runs/adj3_Adam_w3_loss')
for epoch in range(40):
    # dir = './checkpoint/ed_Adam_lr_0.0015_10_w3_h8_epoch{}.pth'.format(epoch)
    dir = './checkpoint/adj3_Addloss_Xflag__Adam_lr_0.0001_1_w3_h1_epoch{}.pth'.format(epoch)
    checkpoint = torch.load(dir)
    loss_a = checkpoint['loss_a']
    # print(epoch, loss_a.item())

    writer1.add_scalar('loss_avarage', loss_a.item(), epoch)
writer1.close()