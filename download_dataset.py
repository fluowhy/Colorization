import torch
import torchvision
from torchvision import datasets


trainset = torchvision.datasets.ImageNet(root="../datasets/imagenet/train", train=True, download=True)
testset = torchvision.datasets.ImageNet(root="../datasets/imagenet/test", train=False, download=True)