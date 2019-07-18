import torch
import torchvision
import random
import os
import skimage
from sklearn.metrics import confusion_matrix
import pandas as pd

from im import *


def save_hyperparamters(names, values, savename):
	df = {}
	for i, name in enumerate(names):
		df[name] = values[i]
	df = pd.DataFrame(data=df)
	df.to_csv("{}.csv".format(savename), index=False)
	return


def numpy2torch(x, device, dtype):
    return torch.tensor(x, device=device, dtype=dtype)


def rgb2bgr(x):
    xbgr = np.zeros(x.shape, dtype=np.uint8)
    xr = x[:, :, 0]
    xg = x[:, :, 1]
    xb = x[:, :, 2]
    xbgr[:, :, 0] = xb
    xbgr[:, :, 1] = xg
    xbgr[:, :, 2] = xr
    return xbgr


def plot_confusion_matrix(y_true, y_pred, labels, normalize=False, title=None, cmap=plt.cm.Blues, dpi=500):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	Modified by M. Romero.
	Original: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	# classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	fig, ax = plt.subplots(figsize=(10, 10))
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=labels,
		   yticklabels=labels,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label'
		   )
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(),
			 rotation=45,
			 ha="right",
			 rotation_mode="anchor"
			 )

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.savefig("figures/cm_{}".format(title), dpi=dpi)
	return ax


def rgb2lab(x):
    """

    Parameters
    ----------
    x : torch.tensor uint8 (N, c, h, w)
        image to be transformed
    Returns
    -------
     : numpy array int8 (N, c, h, w)
        image in lab color space
    """
    return np.transpose(skimage.color.rgb2lab(x.transpose(1, 2).transpose(2, 3).numpy()).astype(np.int8), (0, 3, 1, 2))


def lab2rgb(x):
	"""

	Parameters
	----------
	x : torch.tensor float (N, c, h, w)
		image to be transformed
	Returns
	-------
	 : numpy array uint8 (N, c, h, w)
		image in lab color space
	"""
	for i in range(x.shape[0]):
		x.cpu()
		x[i] = torch.as_tensor(np.transpose(skimage.color.lab2rgb(np.transpose(x[i].cpu().numpy(), (1, 2, 0))), (2, 0, 1)))
	return (x.cpu().numpy()*255.).astype(np.uint8)


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


def load_dataset(debug, N=10, device="cpu", name="cifar10"):
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
	if name == "stl10":
		trainset = torchvision.datasets.STL10(root="../datasets/{}/train".format(name), split="train", download=True)
		testset = torchvision.datasets.STL10(root="../datasets/{}/test".format(name), split="test", download=True)
		unlabeledset = torchvision.datasets.STL10(root="../datasets/{}/unlabeled".format(name), split="unlabeled", download=True)
		if not debug:
			train_tensor = torch.tensor(trainset.data, dtype=torch.float, device="cpu") / 255
			test_tensor = torch.tensor(testset.data, dtype=torch.float, device="cpu") / 255
			unlabeled_tensor = torch.tensor(unlabeledset.data, dtype=torch.float, device="cpu") / 255
		else:
			train_tensor = torch.tensor(trainset.data[:N], dtype=torch.float, device="cpu") / 255.
			test_tensor = torch.tensor(testset.data[:N], dtype=torch.float, device="cpu") / 255.
			unlabeled_tensor = torch.tensor(unlabeledset.data[:N], dtype=torch.float, device="cpu") / 255
		train_lab = colors.rgb_to_lab(train_tensor)
		test_lab = colors.rgb_to_lab(test_tensor)
		unlabeled_lab = colors.rgb_to_lab(unlabeled_tensor)
		train_lab = normalize_lab(train_lab).to(device)
		test_lab = normalize_lab(test_lab).to(device)
		unlabeled_lab = normalize_lab(unlabeled_lab).to(device)
		return train_lab, test_lab, unlabeled_lab
	else:
		trainset = torchvision.datasets.CIFAR10(root="../datasets/{}/train".format(name), train=True, download=True)
		testset = torchvision.datasets.CIFAR10(root="../datasets/{}/test".format(name), train=False, download=True)
		if not debug:
			train_tensor = torch.tensor(trainset.data, dtype=torch.float, device="cpu") / 255
			test_tensor = torch.tensor(testset.data, dtype=torch.float, device="cpu") / 255
		else:
			train_tensor = torch.tensor(trainset.data[:N], dtype=torch.float, device="cpu") / 255.
			test_tensor = torch.tensor(testset.data[:N], dtype=torch.float, device="cpu") / 255.
		train_tensor = train_tensor.transpose(1, -1)
		train_tensor = train_tensor.transpose(-1, -2)
		test_tensor = test_tensor.transpose(1, -1)
		test_tensor = test_tensor.transpose(-1, -2)
		train_lab = colors.rgb_to_lab(train_tensor)
		test_lab = colors.rgb_to_lab(test_tensor)
		train_lab = normalize_lab(train_lab).to(device)
		test_lab = normalize_lab(test_lab).to(device)
		return train_lab, test_lab, 0


def unnormalize_and_lab_2_rgb(x):
	"""
	Transforms an image x from lab to rgb with previous unnormalization.
	Parameters
	----------
	x : torch.tensor, (N, C, w, h)
		Image to unnormalize and transform from lab to rgb.
	Returns
	-------
		Transformed image as numpy array.
	"""
	return colors.lab_to_rgb(unnormalize_lab(x)).cpu().numpy()


def seed_everything(seed=1234):
	"""
	Author: Benjamin Minixhofer
	"""
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	return


def normalize_interval(x, p0, p1=[-1, 1]):
	return (x - p0[0])*(p1[1] - p0[1])/(p1[0] - p0[0]) + p0[1]


def make_folder(dirs=["figures", "models"]):
	for directory in dirs:
		if not os.path.exists(directory):
			os.makedirs(directory)
	return


class AlphaEntropy(torch.nn.Module):
	def __init__(self, sigma_zero, alpha=1.01):
		super(AlphaEntropy, self).__init__()
		self.sigma_zero = sigma_zero
		self.alpha = alpha
		self.epsilon = 1e-8
		self.relu = torch.nn.ReLU()

	def gaussian_gram(self, x, ns, nd):
		if len(x.shape) == 1:
			N = x.shape[0]
			d = 1
			x = x.reshape((N, d))
		else:
			N, d = x.shape
		if ns:
			x = self.normalize_scale(x)
		sigma = self.sigma_zero * N ** (- 1 / (4 + d))
		if nd:
			sigma = sigma * np.sqrt(d)
		A = (- 0.5 * (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=2) / sigma ** 2).exp()
		return A / A.trace()

	def entropy(self, A):
		eigenvalues, _ = torch.symeig(A, eigenvectors=True)
		eigenvalues = self.relu(eigenvalues)
		eigenvalues = eigenvalues / eigenvalues.sum()
		S = eigenvalues.pow(self.alpha).sum().log() / (1 - self.alpha)
		return S

	def normalize_scale(self, x):
		x_mean = x.mean(dim=0)
		x_std = x.std(dim=0)
		return (x - x_mean) / (x_std + self.epsilon)

	def forward(self, x):
		A = self.gaussian_gram(x)
		S = self.entropy(A)
		return S


class MutualInformation(torch.nn.Module):
	def __init__(self, sigma_zero, alpha=1.01, normalize_scale=True, normalize_dimension=True):
		super(MutualInformation, self).__init__()
		self.sigma_zero = sigma_zero
		self.alpha = alpha
		self.relu = torch.nn.ReLU()
		self.alpha_entropy = AlphaEntropy(sigma_zero, alpha)
		self.nor_scale = normalize_scale
		self.nor_dim = normalize_dimension

	def forward(self, x, y):
		A_x = self.alpha_entropy.gaussian_gram(x, self.nor_scale, self.nor_dim)
		A_y = self.alpha_entropy.gaussian_gram(y, self.nor_scale, self.nor_dim)
		S_x = self.alpha_entropy.entropy(A_x)
		S_y = self.alpha_entropy.entropy(A_y)
		AxAy = A_x * A_y
		S_xy = self.alpha_entropy.entropy(AxAy / AxAy.trace())
		return S_x + S_y - S_xy


def getweights(img, binedges, lossweights):
	_, h, w = img.shape
	img_vec = img.reshape(-1)
	img_vec = img_vec * 128.
	img_lossweights = np.zeros(img.shape, dtype='f')
	img_vec_a = img_vec[:h*w]
	binedges_a = binedges[0, ...].reshape(-1)
	binid_a = [binedges_a.flat[np.abs(binedges_a - v).argmin()] for v in img_vec_a]
	img_vec_b = img_vec[h*w:]
	binedges_b = binedges[1, ...].reshape(-1)
	binid_b = [binedges_b.flat[np.abs(binedges_b - v).argmin()] for v in img_vec_b]
	binweights = np.array([lossweights[v1][v2] for v1, v2 in zip(binid_a, binid_b)])
	img_lossweights[0, :, :] = binweights.reshape(h, w)
	img_lossweights[1, :, :] = binweights.reshape(h, w)
	return img_lossweights


class Normalize(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        self.l = [0., 100.]
        self.ab = [- 128., 127.]

    def __call__(self, img):
        return normalize(img[:, 0], self.l).unsqueeze(1), normalize(img[:, 1:], self.ab)


class UnNormalize(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        self.l = [0., 100.]
        self.ab = [- 128., 127.]

    def __call__(self, img):
        img[:, 0] = unnormalize(img[:, 0], self.l)
        img[:, 1:] = unnormalize(img[:, 1:], self.ab)
        return img


class ToType(object):
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def __call__(self, img):
        return img.type(dtype=self.dtype).to(self.device)


class ToLAB(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, img):
        img_lab = np.transpose(skimage.color.rgb2lab(np.transpose(img.cpu().numpy(), (0, 2, 3, 1))), (0, 3, 1, 2))
        return torch.tensor(img_lab, device=self.device, dtype=torch.float)


class ToRGB(object):
    def __init__(self):
        self.f = 1

    def __call__(self, img):
        img = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
        for i in range(img.shape[0]):
            img[i] = skimage.color.lab2rgb(img[i]) * 255
        return np.transpose(img.astype(np.uint8), (0, 3, 1, 2))
