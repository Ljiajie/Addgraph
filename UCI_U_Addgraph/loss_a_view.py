# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: loss_a_view.py
# @time: 2020/04/01
import torch
import numpy as np
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
loss=[]
# writer1 = SummaryWriter('runs/dc_Adam_w3_loss')
for epoch in range(80):
    # dir = './checkpoint/mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(epoch)
    # dir = './checkpoint/l2_mid25_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(epoch)
    dir = './checkpoint/a10_i10_l3_mid100_SPARE_S53_lr_0.001_w_3_epoch{}.pth'.format(epoch)
    checkpoint = torch.load(dir)
    loss_a = checkpoint['loss_a']
    loss.append(loss_a)
    # print(epoch, loss_a.item())
epochs = np.arange(80)+1
plt.plot(epochs, loss, 'ro-')
plt.show()
    # writer1.add_scalar('loss_avarage', loss_a.item(), epoch)
# writer1.close()