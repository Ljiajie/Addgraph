# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: loss_a_view.py
# @time: 2020/04/01
import torch
from tensorboardX import SummaryWriter

writer1 = SummaryWriter('runs/dc_Adam_w3_loss')
for epoch in range(50):
    dir = './checkpoint/dc_Adam_lr_0.032_2_w3_h8_epoch{}.pth'.format(epoch)
    checkpoint = torch.load(dir)
    loss_a = checkpoint['loss_a']
    # print(epoch, loss_a.item())

    writer1.add_scalar('loss_avarage', loss_a.item(), epoch)
writer1.close()