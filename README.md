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
