import torch
import torchvision
from torchvision import datasets


torchvision.datasets.VOCDetection(root="../datasets/vocdet/train", image_set="train", download=True, year="2012")
torchvision.datasets.VOCDetection(root="../datasets/vocdet/test", image_set="val", download=True, year="2012")

torchvision.datasets.STL10(root="../datasets/stl10/train", split="train", download=True)
torchvision.datasets.STL10(root="../datasets/stl10/test", split="test", download=True)
torchvision.datasets.STL10(root="../datasets/stl10/unlabeled", split="unlabeled", download=True)

torchvision.datasets.CIFAR10(root="../datasets/cifar10/train", train=True, download=True)
torchvision.datasets.CIFAR10(root="../datasets/cifar10/test", train=False, download=True)

torchvision.datasets.CIFAR100(root="../datasets/cifar100/train", train=True, download=True)
torchvision.datasets.CIFAR100(root="../datasets/cifar100/train", train=False, download=True)