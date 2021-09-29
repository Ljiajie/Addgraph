# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: Scalar_.py
# @time: 2020/03/18
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/exp1')
for epoch in range(100):
    writer.add_scalar('scalar/test', np.random.rand(), epoch)
    writer.add_scalars('scalar/scalars_test', {'xsinx': epoch * np.sin(epoch), 'xcosx': epoch * np.cos(epoch)}, epoch)

writer.close()