import torch
import torchvision

import matplotlib.pyplot as plt

from tqdm import tqdm
import pytorch_colors as colors

from im import *
from model import *
from utils import *


cuda = True
device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
print(device)

h, w = [64, 64]

transform = [torchvision.transforms.Resize((h, w)), torchvision.transforms.ToTensor()]

trainset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/train", train=True, download=True, transform=torchvision.transforms.Compose(transform))
testset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/test", train=False, download=True, transform=torchvision.transforms.Compose(transform))

N = 100
bs = 50
lr = 2e-4
wd = 0.
epochs = 100

train_tensor = torch.tensor(trainset.data, dtype=torch.float, device="cpu")[:N].float()/255
test_tensor = torch.tensor(testset.data, dtype=torch.float, device="cpu")[:N].float()/255

train_tensor = train_tensor.transpose(1, -1)
train_tensor = train_tensor.transpose(-1, -2)
test_tensor = test_tensor.transpose(1, -1)
test_tensor = test_tensor.transpose(-1, -2)

train_lab = colors.rgb_to_lab(train_tensor)
test_lab = colors.rgb_to_lab(test_tensor)
train_lab = normalize_lab(train_lab).to(device)
test_lab = normalize_lab(test_lab).to(device)

train_lab_set = torch.utils.data.TensorDataset(train_lab)
test_lab_set = torch.utils.data.TensorDataset(test_lab)

trainloader = torch.utils.data.DataLoader(train_lab_set, batch_size=bs, shuffle=True)
testloader = torch.utils.data.DataLoader(test_lab_set, batch_size=bs, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = CONVAE(2, 32, 3, 15).to(device)
print(count_parameters(model))
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss().to(device)


def loss_function(x_out, x, lmi=0.5):
	bs = x.shape[0]
	ch_a_flat_pred = x_out[:, 0].reshape(bs, -1)
	ch_b_flat_pred = x_out[:, 1].reshape(bs, -1)
	loss_a = mse(ch_a_flat_pred, x[:, 0].reshape(bs, -1))
	loss_b = mse(ch_b_flat_pred, x[:, 1].reshape(bs, -1))
	mi_ch_a_b = - mi(ch_a_flat_pred, ch_b_flat_pred, bw=2)
	return loss_a + loss_b - 1/mi_ch_a_b

best_loss = np.inf
for epoch in range(epochs):
	train_loss = train(model, optimizer, trainloader, loss_function)
	test_loss = eval(model, testloader, loss_function)
	if test_loss < best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/color_model.pth")
		best_loss = test_loss
	print_epoch(epoch, train_loss, test_loss)

plot_images(model, train_lab, 4, "train_mse_mi")
plot_images(model, test_lab, 4, "test_mse_mi")

# TODO: add one image processing
# TODO: save best model