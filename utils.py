import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from im import *


def save_or_not(test_loss, best_loss, model, savename):
	if test_loss < best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/{}.pth".format(savename))
		best_loss = test_loss
	return best_loss


def train(model, optimizer, dataloader, loss_function):
	"""

	Parameters
	----------
	model : CONVAE
	optimizer : torch.optim
	dataloader : DataLoader

	Returns
	-------
	loss : float
		One epoch train loss.
	"""
	model.train()
	train_loss = 0
	for idx, batch in enumerate(dataloader):
		batch = batch[0]
		bs = batch.shape[0]
		cL = batch[:, 0]
		cab = batch[:, 1:]
		optimizer.zero_grad()
		output, _ = model(cab)
		loss = loss_function(output, cab)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
	train_loss /= (idx + 1)
	return train_loss


def eval(model, dataloader, loss_function):
    """

    Parameters
    ----------
    # TODO: add loss function to docstring
    loss_fuobject : object
    model : CONVAE
        Model to be evaluated.
    dataloader : DataLoader
        Dataloader to be used.
    Returns
    -------
    eval_loss : float
        Evaluation loss.
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = batch[0]
            bs = batch.shape[0]
            cL = batch[:, 0]
            cab = batch[:, 1:]
            output, _ = model(cab)
            loss = loss_function(output.reshape(bs, -1), cab.reshape(bs, -1))
            eval_loss += loss.item()
        eval_loss /= (idx + 1)
    return eval_loss


def print_epoch(epoch, trainloss, testloss, evalloss=None):
	"""

	Parameters
	----------
	epoch : int
		Train epoch.
	trainloss : float:
		Train loss.
	testloss : float
		Test loss.
	evalloss : float
		Eval loss.

	Returns
	-------
	None
	"""
	print("Epoch {} Train Loss {:.3f} Test Loss {:.3f}".format(epoch, trainloss, testloss)) if evalloss == None else print(
		"Epoch {} Train Loss {:.3f} Eval Loss {:.3f} Test Loss {:.3f}".format(epoch, trainloss, evalloss, testloss))
	return


def count_parameters(model):
	# TODO: add docstring
	"""

	Parameters
	----------
	model

	Returns
	-------

	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_images(model, test_data, n, title="test"):
	"""
	Evaluates the model and plots n random images from test.
	Returns
	-------

	"""
	model.eval()
	idx = np.random.choice(np.arange(0, test_data.shape[0]), size=n, replace=False)
	with torch.no_grad():
		ab, _ = model(test_data[idx, 1:])
		ab = ab.cpu()
	real_images = test_data[idx].cpu()
	colorized_images = torch.zeros((n, 3, ab.shape[2], ab.shape[2]))
	colorized_images[:, 0] = real_images[:, 0]
	colorized_images[:, 1] = ab[:, 0]
	colorized_images[:, 2] = ab[:, 1]
	colorized_images = unnormalize_lab(colorized_images)
	real_images = unnormalize_lab(real_images)
	colorized_images = colors.lab_to_rgb(colorized_images).numpy()
	real_images = colors.lab_to_rgb(real_images).numpy()
	colorized_images = np.transpose(colorized_images, (0, 2, 3, 1))
	real_images = np.transpose(real_images, (0, 2, 3, 1))
	fs = 5
	fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(fs, n*fs))
	for i in range(n):
		ax[i, 0].imshow(real_images[i])
		ax[i, 1].imshow(colorized_images[i])
	ax[0, 0].set_title("real {}".format(title))
	ax[0, 1].set_title("colorized {}".format(title))
	plt.tight_layout()
	plt.savefig("figures/colorized_{}".format(title), dpi=400)
	plt.show()
	return


class GMMLoss(torch.nn.Module):
	def __init__(self, ld):
		super(GMMLoss, self).__init__()
		self.cte = np.sqrt(2 * np.pi)**ld

	def forward(self, mu, log_s2, w, z):
		r = (- 0.5 * ((z.unsqueeze(-1) - mu).pow(2) / log_s2.exp()).sum(dim=1)).exp()
		det = log_s2.exp().prod(dim=1).sqrt() * self.cte
		pr = (r / det * w).sum(dim=1)
		return (- torch.log(pr)).mean()


def train_gmm(model, optimizer, dataloader, loss_function):
	"""

	Parameters
	----------
	model : torch.nn.module
	optimizer : torch.optim
	dataloader : DataLoader

	Returns
	-------
	loss : float
		One epoch train loss.
	"""
	model.train()
	train_loss = 0
	for idx, batch in enumerate(dataloader):
		x, z = batch
		optimizer.zero_grad()
		mu, log_s2, w = model(x)
		loss = loss_function(mu, log_s2, w, z)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
	train_loss /= (idx + 1)
	return train_loss


def eval_gmm(model, dataloader, loss_function):
	"""

	Parameters
	----------
	# TODO: add loss function to docstring
	loss_fuobject : object
	model : CONVAE
		Model to be evaluated.
	dataloader : DataLoader
		Dataloader to be used.
	Returns
	-------
	eval_loss : float
		Evaluation loss.
	"""
	model.eval()
	eval_loss = 0
	with torch.no_grad():
		for idx, batch in enumerate(dataloader):
			x, z = batch
			mu, log_s2, w = model(x)
			loss = loss_function(mu, log_s2, w, z)
			eval_loss += loss.item()
		eval_loss /= (idx + 1)
	return eval_loss


def gmmloss(mu, log_s2, w, z):
	r = (- 0.5 * ((z.unsqueeze(-1) - mu).pow(2) / log_s2.exp()).sum(dim=1)).exp()
	det = log_s2.exp().prod(dim=1).sqrt()
	pr = (r / det * w).sum(dim=1)
	return (- torch.log(pr)).mean()


def load_dataset(N, device="cpu", all=False):
    """
    load data from Cifar10.
    Parameters
    ----------
    N
    all

    Returns
    -------
    Torch tensors from train and test sets.
    """

    trainset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/train", train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root="../datasets/cifar10/test", train=False, download=True)
    if all:
        train_tensor = torch.tensor(trainset.data, dtype=torch.float, device="cpu") / 255
        test_tensor = torch.tensor(testset.data, dtype=torch.float, device="cpu") / 255
    else:
        train_tensor = torch.tensor(trainset.data[:N], dtype=torch.float, device="cpu") / 255
        test_tensor = torch.tensor(testset.data[:N], dtype=torch.float, device="cpu") / 255
    train_tensor = train_tensor.transpose(1, -1)
    train_tensor = train_tensor.transpose(-1, -2)
    test_tensor = test_tensor.transpose(1, -1)
    test_tensor = test_tensor.transpose(-1, -2)
    train_lab = colors.rgb_to_lab(train_tensor)
    test_lab = colors.rgb_to_lab(test_tensor)
    train_lab = normalize_lab(train_lab).to(device)
    test_lab = normalize_lab(test_lab).to(device)
    return train_lab, test_lab
