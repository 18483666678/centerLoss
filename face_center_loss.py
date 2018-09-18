import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from face_datasets import train_data, train_loader
from centerlossDemo import CenterLoss


class FaceNet(nn.Module):

    def __init__(self):
        super(FaceNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout2d(),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128 * 4),
            nn.PReLU(),
            # nn.Dropout(),
            nn.Linear(128 * 4, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(256, 2),
            nn.BatchNorm1d(2),
            nn.PReLU(),
        )
        self.fc2 = nn.Linear(2, 6)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1, 128 * 4 * 4)
        feat = self.fc1(y)
        y = self.fc2(feat)
        return feat, F.log_softmax(y, dim=1)


def main():
    facenet = FaceNet().cuda()
    print(facenet)
    sloss_fn = nn.CrossEntropyLoss()
    opt_soft = optim.Adam(facenet.parameters(), lr=0.0001)
    closs_fn = CenterLoss(6, 2, use_gpu=True)
    opt_cent = optim.Adam(closs_fn.parameters(), lr=0.0001)
    # facenet.load_state_dict(torch.load("face_center_loss.pkl"))

    for epoch in range(1000):
        train(facenet, sloss_fn, opt_soft, closs_fn, opt_cent)
        test(facenet, sloss_fn, closs_fn)


def train(facenet, sloss_fn, opt_soft, closs_fn, opt_cent):
    facenet.train()
    features = []
    labels = []
    for step, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()
        feat, out = facenet(img)

        # pred = torch.max(out, 1)[1]
        # print("pred", pred, pred.shape)
        # print("Accuracy", (pred == label).sum().float() / float(label.size(0)))

        sloss = sloss_fn(out, label)
        closs = closs_fn(feat, label)
        loss = sloss + closs
        print("sloss:", sloss, "closs:", closs, "loss:", loss)
        opt_soft.zero_grad()
        opt_cent.zero_grad()
        loss.backward()
        opt_soft.step()
        opt_cent.step()
        features.append(feat)
        labels.append(label)

    torch.save(facenet.state_dict(), "face_center_loss.pkl")
    features = torch.cat(features).data.cpu().numpy()
    labels = torch.cat(labels).data.cpu().numpy()


def test(facenet, sloss_fn, closs_fn):
    facenet.eval()
    features = []
    labels = []
    with torch.no_grad():
        for step, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            feat, out = facenet(img)

            pred = torch.max(out, 1)[1]
            # print("pred", pred, pred.shape)
            print(label.size(0))
            print("Accuracy", (pred == label).sum().float() / float(label.size(0)))

            sloss = sloss_fn(out, label)
            closs = closs_fn(feat, label)
            loss = sloss + closs
            print("sloss:", sloss, "closs:", closs, "loss:", loss)

            features.append(feat)
            labels.append(label)

    features = torch.cat(features).data.cpu().numpy()
    labels = torch.cat(labels).data.cpu().numpy()
    visualize(features, labels)


def visualize(features, labels):
    plt.ion()
    plt.clf()
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(6):
        plt.plot(features[i == labels, 0], features[i == labels, 1], ".", c=color[i])
    plt.legend(['0', '1', '2', '3', '4', '5'], loc="upper right")
    plt.pause(0.01)
    plt.ioff()


if __name__ == '__main__':
    main()
