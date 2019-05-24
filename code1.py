import torch
import torchvision
import matplotlib.pyplot as plt

nt = 4

image_data = torchvision.datasets.CIFAR100("../datasets/cifar100", download=True, transform=torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(image_data, batch_size=4, shuffle=True, num_workers=nt)