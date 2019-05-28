import torch
import torchvision

import matplotlib.pyplot as plt

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

N = 2000
bs = 50
lr = 2e-4
wd = 0.
epochs = 100
ld = 2 # latent space dimension
k = 3 # colorizations

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

model = CONVAE(2, 32, 3, ld).to(device)
print(count_parameters(model))
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss().to(device)
gmml = GMMLoss(ld)


def loss_function(x_out, x, lmi=0.5):
	bs = x.shape[0]
	ch_a_flat_pred = x_out[:, 0].reshape(bs, -1)
	ch_b_flat_pred = x_out[:, 1].reshape(bs, -1)
	loss_a = mse(ch_a_flat_pred, x[:, 0].reshape(bs, -1))
	loss_b = mse(ch_b_flat_pred, x[:, 1].reshape(bs, -1))
	# mi_ch_a_b = - mi(ch_a_flat_pred, ch_b_flat_pred, bw=2)
	return loss_a + loss_b  # + lmi * mi_ch_a_b

gmm = CONVGMM(1, 32, 5, 1024, k, ld).to(device)
optimizer_gmm = torch.optim.Adam(gmm.parameters(), lr=lr, weight_decay=wd)

for epoch in range(epochs):
	model.train()
	gmm.train()
	train_loss_ae = 0
	train_loss_gmm = 0
	for idx, batch in enumerate(trainloader):
		batch = batch[0]
		cL = batch[:, 0].unsqueeze(1)
		cab = batch[:, 1:]
		optimizer.zero_grad()
		optimizer_gmm.zero_grad()
		output, z = model(cab)
		z = z.detach().clone()
		mu, log_s2, w = gmm(cL)
		loss_ae = loss_function(output, cab)
		loss_gmm = gmml(mu, log_s2, w, z)
		loss_ae.backward()
		loss_gmm.backward()
		optimizer.step()
		optimizer_gmm.step()
		train_loss_ae += loss_ae.item()
		train_loss_gmm += loss_gmm.item()
	train_loss_ae /= (idx + 1)
	train_loss_gmm /= (idx + 1)
	print("Epoch {} AE train loss {:.3f} GMM train loss {:.3f}".format(epoch, train_loss_ae, train_loss_gmm))

model.eval()
gmm.eval()
img_lab = torch.zeros((k, test_lab.shape[0], 3, test_lab.shape[2], test_lab.shape[2]))
img_rgb = np.zeros((k, test_lab.shape[0], 3, test_lab.shape[2], test_lab.shape[2]))
with torch.no_grad():
	z, _, _ = gmm(test_lab[:, 0].unsqueeze(1))
	for i in range(k):
		ab = model.decode(z[:, :, i].reshape((z.shape[0], z.shape[1], 1, 1)))
		img_lab[i, :, 1] = ab[:, 0]
		img_lab[i, :, 2] = ab[:, 1]
		img_lab[i, :, 0] = test_lab[:, 0]
		img_rgb[i] = unnormalize_and_lab_2_rgb(img_lab[i])
		plt.imshow(np.transpose(img_rgb[i, 0], (1, 2, 0)))
		plt.savefig("figures/colorized_{}".format(i), dpi=400)

# TODO: add one image processing
# TODO: save best model
# TODO try MI between L,a and L,b
# TODO: add real image
# TODO fix gmm high dim