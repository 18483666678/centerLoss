"""
https://blog.csdn.net/hao5335156/article/details/80607732
（三）PyTorch学习笔记——softmax和log_softmax的区别、CrossEntropyLoss() 与 NLLLoss() 的区别、log似然代价函数
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data = autograd.Variable(torch.FloatTensor([1.0, 2.0, 3.0]))
log_softmax = F.log_softmax(data, dim=0)
print(log_softmax)

softmax = F.softmax(data, dim=0)
print(softmax)

np_softmax = softmax.data.numpy()
log_np_softmax = np.log(np_softmax)
print(log_np_softmax)
