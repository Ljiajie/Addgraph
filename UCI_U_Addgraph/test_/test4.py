# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: test4.py
# @time: 2020/03/18
import torch
from tensorboardX import SummaryWriter

writer = SummaryWriter()
x = torch.FloatTensor([100])
y = torch.FloatTensor([500])

for epoch in range(100):
    x /= 1.5
    y /= 1.5
    loss = y - x
    print(loss)
    writer.add_histogram('zz/x', x, epoch)
    writer.add_histogram('zz/y', y, epoch)
    writer.add_scalar('data/x', x, epoch)
    writer.add_scalar('data/y', y, epoch)
    writer.add_scalar('data/loss', loss, epoch)
    writer.add_scalars('data/scalar_group', {'x': x,
                                             'y': y,
                                             'loss': loss}, epoch)
    writer.add_text('zz/text', 'zz: this is epoch ' + str(epoch), epoch)

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./test.json")
writer.close()