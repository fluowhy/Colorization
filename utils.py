import torch
import torchvision
import random
import os
import skimage
import cv2
import pandas as pd

from im import *


def read_hyperparameters(savename):
	params_df = pd.read_csv(savename)
	names = params_df.columns
	params = {}
	for name in names:
		params[name] = int(params_df[name].values[0])
	return params


def save_hyperparamters(names, values, savename):
	df = {}
	for i, name in enumerate(names):
		df[name] = [values[i]]
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
	def __init__(self, sigma_zero=2, alpha=1.01, normalize_scale=True, normalize_dimension=True):
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
	def __init__(self, device):
		self.cte = torch.tensor([100, 128, 128], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

	def __call__(self, x):
		x = x / self.cte
		return x[:, 0].unsqueeze(1), x[:, 1:]


class UnNormalize(object):
	def __init__(self, device):
		self.cte = torch.tensor([1, 0.5, 0.5], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
		self.cte_sum = torch.tensor([0, 1, 1], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

	def __call__(self, x):
		return (x + self.cte_sum) * self.cte * 255


class ToType(object):
	def __init__(self, dtype, device):
		self.dtype = dtype
		self.device = device

	def __call__(self, img):
		return img.type(dtype=self.dtype).to(self.device)


class GreyTransform(object):
	def __init__(self, dtype, device):
		self.dtype = dtype
		self.device = device

	def __call__(self, x):
		return (x.type(dtype=self.dtype).to(self.device) / 255).unsqueeze(1)


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
