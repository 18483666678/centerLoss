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
