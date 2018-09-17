import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image

# copy from gen_face_data.py
label_path = "./person/list_bbox_celeba.txt"
img_path = "./person"
save_path = "../person_face"
face_size = 64
COUNT = 50

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = torchvision.datasets.ImageFolder(save_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)

if __name__ == '__main__':
    print(train_data.classes)
    print(train_data.class_to_idx)
    print(train_data[0][0])
    plt.ion()
    for step, (img, label) in enumerate(train_loader):
        print(step, img.shape, label.shape)
        grid = utils.make_grid(img)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        cls = [train_data.classes[lab] for lab in label]
        print(cls)
        plt.pause(3)
