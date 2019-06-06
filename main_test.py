import torch
import torchvision

import matplotlib.pyplot as plt

from tqdm import tqdm
import pytorch_colors as colors

from im import *
from model import *
from utils import *
from vae import *
from mdn import *

cuda = False
device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
print(device)

h, w = [64, 64]
N = 10
bs = 50
lr = 2e-4
wd = 0.
epochs = 100

train_lab, test_lab = load_dataset(N, device)
vae = VAE().to(device)
print(count_parameters(vae))
optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss().to(device)

for epoch in range(epochs):
