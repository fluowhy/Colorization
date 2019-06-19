import torchvision

name = "stl10"

if name == "vocdet":
    trainset = torchvision.datasets.VOCDetection(root="../datasets/vocdet/train", image_set="train", download=True, year="2012") # trainnval
    testset = torchvision.datasets.VOCDetection(root="../datasets/vocdet/test", image_set="val", download=True, year="2012")
elif name == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/train", train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/test", train=False, download=True)
elif name == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root="../datasets/cifar100/train", train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root="../datasets/cifar100/train", train=False, download=True)
elif name == "stl10":
    trainset = torchvision.datasets.STL10(root="../datasets/stl10/train", split="train", download=True)
    testset = torchvision.datasets.STL10(root="../datasets/stl10/test", split="test", download=True)
    unlabeledset = torchvision.datasets.STL10(root="../datasets/stl10/unlabeled", split="unlabeled", download=True)
