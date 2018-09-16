from __future__ import print_function  # make python 2 like python 3 syntax
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class CenterLoss(nn.Module):

    def __init__(self, classes_dim, feature_dim, use_gpu=False):
        super(CenterLoss, self).__init__()
        self.classes_dim = classes_dim
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(classes_dim, feature_dim).to("cuda"))
        else:
            self.centers = nn.Parameter(torch.randn(classes_dim, feature_dim))

    def forward(self, feat, label):
        scent = self.centers.index_select(0, label)
        # print("scent\n", scent)
        if self.use_gpu:
            counts = torch.histc(label.cpu().float(), bins=self.classes_dim, min=0, max=self.classes_dim).to("cuda")
        else:
            counts = torch.histc(label.float(), bins=self.classes_dim, min=0, max=self.classes_dim)
        # print("counts\n", counts)
        scounts = counts.index_select(0, label)
        # print("scounts\n", scounts)
        loss = ((feat - scent).pow(2).sum(1) / scounts).sum() / label.size(0)
        return loss


# data = torch.Tensor([[1, 2], [2, 3], [3, 6]])
# data = torch.zeros(3, 2).to("cuda")
# label = torch.Tensor([0, 0, 1]).to("cuda")
#
# center_loss = CenterLoss(10, 2, use_gpu=True)
# print(list(center_loss.parameters()))
# loss = center_loss(data, label)
# print(loss)
# print("end")
# exit()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, 2)
        self.ip2 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return F.log_softmax(ip2, dim=1), ip1


def train(args, model, device, train_loader, optimizer, epoch, center_loss, opt_closs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape, target.shape, len(train_loader))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        opt_closs.zero_grad()
        output, center_out = model(data)
        loss = F.nll_loss(output, target) + center_loss(center_out, target)
        # print(center_loss(center_out, target))
        loss.backward()
        optimizer.step()
        opt_closs.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # torch.save(model.state_dict(), "train_mnist.pkl")


def test(args, model, device, test_loader, center_loss, opt_closs):
    model.eval()
    test_loss = 0
    correct = 0
    plt.ion()
    plt.clf()
    center = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, center_out = model(data)
            test_loss += F.nll_loss(output, target).item() + center_loss(center_out, target)  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(data.shape, data[0], target.shape, target[0])
            # print("out:", output.shape, output[0])
            # print("pred", pred.shape, pred[0])
            # print("xxxx\n", target.view_as(pred))
            # print("zzzz>>>\n", output.max(1, keepdim=True))
            center.append(center_out)
            labels.append(target)
    center_out = torch.cat(center, 0)
    target = torch.cat(labels, 0)
    center = center_out.data.cpu().numpy()
    labels = target.data.cpu().numpy()
    print(center_out.shape)
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    for i in range(10):
        plt.plot(center[labels == i, 0], center[labels == i, 1], ".", c=c[i])
        # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        # plt.pause(1)
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # plt.draw()
    plt.show()
    plt.pause(1)
    # visualize(center_out.data.cpu().numpy(), target.data.cpu().numpy())
    plt.ioff()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def visualize(feat, labels):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.xlim(xmin=-5, xmax=5)
    plt.ylim(ymin=-5, ymax=5)
    plt.draw()
    plt.pause(0.1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    print(model)
    # model.load_state_dict(torch.load("train_mnist.pkl"))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    center_loss = CenterLoss(10, 2, use_gpu=True)
    opt_closs = optim.Adam(center_loss.parameters(), lr=0.0001)
    print(list(center_loss.parameters()))

    for epoch in range(1, 10000000):
        train(args, model, device, train_loader, optimizer, epoch, center_loss, opt_closs)
        test(args, model, device, test_loader, center_loss, opt_closs)


if __name__ == '__main__':
    main()
