Author: lichen_zeng@sina.cn
Date: 20180913
Subject: Note the process of center loss debug


Reference:
https://github.com/pytorch/examples/blob/master/mnist/main.py
https://github.com/jxgu1016/MNIST_center_loss_pytorch/blob/master/MNIST_with_centerloss.py


Debug key point:
======
1, 需要在中间某个环节输出 2维 的坐标点(x, y)，当然也可以用三维的坐标点。
    self.fc1 = nn.Linear(320, 50)
    self.fc3 = nn.Linear(50, 2)
    self.fc2 = nn.Linear(2, 10)

2, 可以借助 列表 来批量累计数据，然后一次性显示大量数据。
    center = []
    labels = []
        center.append(center_out)
        labels.append(target)
    center_out = torch.cat(center, 0)
    target = torch.cat(labels, 0)

3, 以下是 MNIST显示2维坐标数据 最为关键的代码 -- 需要根据labels分类将相应的点分开显示
    for i in range(10):
        plt.plot(center[labels == i, 0], center[labels == i, 1], ".", c=c[i])

4, 在pytorch中 softmax 和 log_softmax 的区别？
    return F.softmax(x, dim=1)
    return F.log_softmax(x, dim=1)

5, 在pytorch中可以按照如下方法创建 one-hot 形式。
    target = torch.zeros(target.shape[0], 10).scatter_(1, target.view(target.shape[0], -1), 1)

6, 为什么我将 log_softmax 修改为 softmax ，NLLLoss 修改为 mse_loss（同时target需要弄成one-hot形式），结果就训练不出来了呢？
    其实在Tensorflow中就是这样处理的，应该是哪里没有弄对，暂时备忘下！

7, pytorch 中的模型保存和导入
    model.load_state_dict(torch.load("train_mnist.pkl"))
    torch.save(model.state_dict(), "train_mnist.pkl")

8, 在pytorch中常用的导入包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

9, pytorch的网络定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x

10, 在pytorch中，通过nn.Sequential()可以很方便定义网络结构 -- 注意：中间的各个层需要用 逗号 隔开
    self.conv = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
        nn.PReLU(),
        nn.MaxPool2d(kernel_size=2),
    )

11, torchvision中的transforms处理图片数据非常方便，切记Compose([])中的 []，以及需要用 逗号 隔开，被处理的的图片维度会变化 (h, w, c) --> (c, h, w)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(...)
    ])

12, 对于某些Hello world级别的常用数据集，pytorch已经集成。
train_data = torchvision.datasets.MNIST(root="../MNIST_data", train=True, transform=transform, download=True)

以下方法可以高效的取出一批数据
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

以下是pytorch的数据集定义固定格式
class MNIST(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

    def __getitem__(self, index):
        return img, target

    def __len__(self):
        return len()

可以通过以下方法取出数据
for img, label in train_loader:
    print(img.shape, label.shape)


13, pytorch 固定的训练模式
（1）创建网络实例
（2）选择损失函数
（3）选择优化函数（优化其中的参数，设置学习率）
（4）批量取出（数据，标签）来训练网络
（5）计算损失
（6）梯度清零
（7）反向传播损失
（8）更新梯度
    net = Net()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=0.0001)
    for step, (img, label) in enumerate(train_loader):
        output = net(img)
        loss = loss_func(output, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

