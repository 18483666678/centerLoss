import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from matplotlib import pyplot as plt
import numpy as np


class CenterLoss3(nn.Module):

    def __init__(self, classes_dim, feature_dim, use_gpu=False):
        super(CenterLoss3, self).__init__()
        self.classes_dim = classes_dim
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.classes_dim, self.feature_dim).to("cuda"))
        else:
            self.centers = nn.Parameter(torch.randn(self.classes_dim, self.feature_dim))

    def forward(self, feat, label):
        scent = self.centers.index_select(0, label)
        # print("scent\n", scent)
        if self.use_gpu:
            counts = (torch.histc(label.cpu().float(), bins=self.classes_dim, min=0, max=self.classes_dim) + 1).to("cuda")
        else:
            counts = torch.histc(label.float(), bins=self.classes_dim, min=0, max=self.classes_dim) + 1
        # print("counts\n", counts)
        scounts = counts.index_select(0, label)
        # print("scounts\n", scounts)
        loss = (torch.sqrt((feat - scent).pow(2).sum(1) / scounts)).sum()
        # distance = scent.dist(scent)
        # loss = distance / label.size(0)

        return loss


class CenterLoss2(nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0, use_gpu=False):
        super(CenterLoss2, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.use_cuda = False

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, feat, y):
        # if self.use_cuda:
        #     hist = Variable(
        #         torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1).cuda()
        # else:
        #     hist = Variable(torch.histc(y.data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1)
        if self.use_gpu:
            hist = (torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1).cuda()
        else:
            hist = (torch.histc(y.data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1)

        centers_count = hist.index_select(0, y.long())  # 计算每个类别对应的数目

        batch_size = feat.size()[0]
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,
                                                                                                    feat.size()[1]))
        centers_pred = self.centers.index_select(0, y.long())
        diff = feat - centers_pred
        loss = self.loss_weight * 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()
        return loss

    # def cuda(self, device_id=None):
    #     self.use_cuda = True
    #     return self._apply(lambda t: t.cuda(device_id))


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.PReLU(),
            nn.Linear(120, 84),
            nn.PReLU(),
            nn.Linear(84, 2),
        )
        self.y = nn.Linear(2, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        y = self.y(x)
        return x, F.log_softmax(y, 1)


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = torchvision.datasets.MNIST(root="../MNIST_data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

test_data = torchvision.datasets.MNIST(root="../MNIST_data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

wight_closs = 1.0
if __name__ == '__main__':
    net = Net().to("cuda")
    print(net)
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=0.0001)

    # net.load_state_dict(torch.load("centerloss.pkl"))
    center_loss = CenterLoss3(10, 2, use_gpu=True)
    # center_loss = center_loss.cuda(0)
    opt_closs = optim.Adam(center_loss.parameters(), lr=0.001)

    # x = torch.rand(1, 1, 28, 28)
    # print(x.shape)
    # y = net(x)
    # print(y.shape)
    # print(len(train_data))
    # print(len(train_loader))
    for i in range(100000):
        feature = []
        labels = []
        for step, (img, label) in enumerate(train_loader):
            # print(img.shape, label.shape)
            img = img.to("cuda")
            label = label.to("cuda")
            ft, output = net(img)
            # print(output.shape, ft.shape)
            # print(type(ft))
            loss = loss_func(output, label) + center_loss(ft, label) * wight_closs
            print(loss)
            opt.zero_grad()
            opt_closs.zero_grad()
            loss.backward()
            opt.step()
            for param in center_loss.parameters():
                param.grad.data *= (1. / wight_closs)
            opt_closs.step()
            feature.append(ft)
            labels.append(label)

        # torch.save(net.state_dict(), "centerloss.pkl")
        fea = torch.cat(feature, 0).cpu().data.numpy()
        lab = torch.cat(labels, 0).cpu().data.numpy()

        plt.ion()
        plt.clf()
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for c in range(10):
            plt.plot(fea[c == lab, 0], fea[c == lab, 1], '.', c=color[c])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.pause(1)

        # for img1, label1 in test_loader:
        #     grid = utils.make_grid(img1)
        #     plt.clf()
        #     plt.imshow(grid.numpy().transpose((1, 2, 0)))
        #     plt.show()
        #     output1 = net(img1)
        #     predict = torch.max(output1, 1)[1]
        #     # print(predict.shape, label1.shape)
        #     acc = float((predict == label1).sum().item()) / float(label1.size(0))
        #     print(acc)
        #     plt.pause(1)
        #     break
        #
