# _*_ coding:utf-8 _*_
# @author:Jiajie Lin
# @file: test_save.py
# @time: 2020/04/02
import numpy as np
c=[]
a=3
b=np.array([2,3])
d=np.array([1,2])
c.append(b)
c.append(d)
np.savez("data.npz",data=a,label=c)
data=np.load("data.npz")
a,b=data['data'],data['label']
print(a,'\n',b)