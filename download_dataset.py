import torch
import torchvision


trainset = torchvision.datasets.ImageNet(root="../datasets/imagenet/train", train=True, download=True)
testset = torchvision.datasets.ImageNet(root="../datasets/imagenet/test", train=False, download=True)