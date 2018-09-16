import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CenterLoss(nn.Module):

    def __init__(self, cls_dim, feat_dim, use_gpu=False):
        super(CenterLoss, self).__init__()
        self.cls_dim = cls_dim
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.cls_dim, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.cls_dim, self.feat_dim))

    def forward(self, feat, label):
        scent = self.centers.index_select(0, label)
        if self.use_gpu:
            hist = torch.histc(label.cpu().float(), bins=self.cls_dim, min=0, max=self.cls_dim).cuda()
        else:
            hist = torch.histc(label.float(), bins=self.cls_dim, min=0, max=self.cls_dim)
        count = hist.index_select(0, label)
        loss = (torch.sqrt((feat - scent).pow(2).sum(1) / count)).sum() / label.size(0)
        return loss


class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, 3)
        )
        self.fc2 = nn.Linear(3, 10)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1, 16 * 5 * 5)
        feat = self.fc(y)
        y = self.fc2(feat)
        return feat, F.log_softmax(y, 1)


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = torchvision.datasets.MNIST("../MNIST_data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

test_data = torchvision.datasets.MNIST("../MNIST_data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)

if __name__ == '__main__':
    cnn = CNNNet().cuda()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(cnn.parameters(), lr=0.0001)
    center_loss = CenterLoss(10, 3, use_gpu=True)
    opt_cent = optim.Adam(center_loss.parameters(), lr=0.0001)

    # cnn.load_state_dict(torch.load("centerlossdemo.pkl"))

    for i in range(100):
        features = []
        labels = []
        # Train stage
        for step, (img, label) in enumerate(train_loader):
            img = img.to("cuda")
            label = label.to("cuda")
            feat, out = cnn(img)
            # pred = torch.max(out, 1)[1]
            # print("pred", pred, pred.shape)
            # print(label.size(0))
            # print("Accuracy", (pred == label).sum().float() / float(label.size(0)))

            sloss = loss_func(out, label)
            closs = center_loss(feat, label)
            loss = sloss + closs
            print("sloss:", sloss, "\tcloss:", closs)
            opt.zero_grad()
            opt_cent.zero_grad()
            loss.backward()
            opt.step()
            opt_cent.step()
            features.append(feat)
            labels.append(label)

        # torch.save(cnn.state_dict(), "centerlossdemo.pkl")
        features = torch.cat(features).cpu().data.numpy()
        labels = torch.cat(labels).cpu().data.numpy()
        plt.ion()
        plt.clf()
        # fig = plt.figure()
        # ax = Axes3D(fig)
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i in range(10):
            plt.plot(features[i == labels, 0], features[i == labels, 2], ".", c=color[i])
            # ax.scatter(features[i == labels, 0], features[i == labels, 1], features[i == labels, 2], c=color(i),
            #            marker='.', s=50, label='')
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.show()
        plt.pause(1)
        plt.ioff()

        # # Test stage
        # with torch.no_grad():
        #     for img, label in test_loader:
        #         out = cnn(img)
        #         predict = torch.max(out, 1)[1]
        #         print("Accuracy", (predict == label).sum().float() / float(label.size(0)))
