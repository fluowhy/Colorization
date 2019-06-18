import torch
import torchvision
from torchvision import datasets


trainset = torchvision.datasets.VOCDetection(root="../datasets/vocdet/train", image_set="train", download=True, year="2012")
testset = torchvision.datasets.VOCDetection(root="../datasets/vocdet/test", image_set="val", download=True, year="2012")

trainset = torchvision.datasets.STL10(root="../datasets/stl10/train", split="train", download=True)
testset = torchvision.datasets.STL10(root="../datasets/stl10/test", split="test", download=True)
unlabeled = torchvision.datasets.STL10(root="../datasets/stl10/unlabeled", split="unlabeled", download=True)