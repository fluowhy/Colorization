import torch
import numpy as np
import matplotlib.pyplot as plt

from im import *


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
	fs = 8
	fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(fs, n*fs))
	for i in range(n):
		ax[i, 0].imshow(real_images[i])
		ax[i, 1].imshow(colorized_images[i])
	ax[0, 0].set_title("real {}".format(title))
	ax[0, 1].set_title("colorized {}".format(title))
	plt.show()
	return