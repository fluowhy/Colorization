import torchvision
from sklearn.model_selection import train_test_split
import numpy as np


trainset = torchvision.datasets.STL10(root="../datasets/stl10", split="train", download=False)

indexes = np.arange(trainset.data.shape[0])

train_idx, val_idx = train_test_split(indexes, test_size=0.1, stratify=trainset.labels)

np.save("train_idx", train_idx)
np.save("val_idx", val_idx)