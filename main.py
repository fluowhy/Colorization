import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import pytorch_colors as colors

from im import *
from model import *


argscuda = True
device = torch.device("cuda:0" if argscuda and torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0., 0., 0.), (1., 1., 1.))])

trainset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/train", train=True, download=True)#, transform=transform)
testset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/test", train=False, download=True)#, transform=transform)

train_tensor = torch.tensor(trainset.data).float().to(device)/255
test_tensor = torch.tensor(testset.data).float().to(device)/255

train_tensor = train_tensor.transpose(1, -1)
train_tensor = train_tensor.transpose(-1, -2)
test_tensor = test_tensor.transpose(1, -1)
test_tensor = test_tensor.transpose(-1, -2)

train_lab = colors.rgb_to_lab(train_tensor)
test_lab = colors.rgb_to_lab(test_tensor)
train_lab = normalize_lab(train_lab)
test_lab = normalize_lab(test_lab)

train_lab_set = torch.utils.data.TensorDataset(train_lab[:, 0], train_lab[:, 1], train_lab[:, 2])
test_lab_set = torch.utils.data.TensorDataset(test_lab[:, 0], test_lab[:, 1], test_lab[:, 2])

trainloader = torch.utils.data.DataLoader(train_lab_set, batch_size=4, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_lab_set, batch_size=4, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

lr = 2e-4
wd = 0.
epochs = 2

model = CONVAE().to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
bce = torch.nn.BCELoss()

for epoch in range(epochs):
	model.train()
	train_loss = 0
	for idx, (img_gray, img_a, img_b) in enumerate(trainloader):
		print(idx)
		optimizer.zero_grad()
		output, _ = model(img_gray)
		loss = bce(output[:, 0], img_a) + bce(output[:, 1], img_b)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		print(train_loss)
	train_loss /= (idx + 1)
	"""
	test_loss = 0
	model.eval()
	with torch.no_grad():
		for idx, (batch, y_true, batch_lengths) in enumerate(tqdm(test_loader)):
			output, hn = model(batch, batch_lengths)
			output = output[torch.arange(output.shape[0]), batch_lengths - 1, :]
			loss = cel(output, y_true)
			test_loss += loss.item()
	test_loss /= (idx + 1)
	tf = time.time()
	print("Epoch {:03d} | Train loss {:.3f} | Test loss {:.3f} | Time {:.2f} min.".format(epoch, train_loss, test_loss, (tf - ti)/60))
	if test_loss<best_loss and test_loss>0:
		print("Saving")
		torch.save(model.state_dict(), "models/lstm_plasticc.pth")
		best_loss = test_loss
		best_epoch = epoch
		counter = 0
	else:
		counter += 1
	if counter==patience:
		break
	else:
		0
	"""