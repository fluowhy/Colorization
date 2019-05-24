import torch
import torchvision
import matplotlib.pyplot as plt

from model import *

nt = 4

h, w = [64, 64]

cuda = False
device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")

transforms = [torchvision.transforms.Resize((h, w)), torchvision.transforms.ToTensor()]

image_data = torchvision.datasets.CIFAR100("../datasets/cifar100", download=True, transform=torchvision.transforms.Compose(transforms))
data_loader = torch.utils.data.DataLoader(image_data, batch_size=4, shuffle=True)

model = CNNAE(3, 4, 2).to(device)
plt.figure()
for i, batch in enumerate(data_loader):
    x, y = batch
    l = model(x)
    d = 0

