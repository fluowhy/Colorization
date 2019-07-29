from tqdm import tqdm
import argparse

from divcolor import *


def vae_loss(mu, logvar, pred, gt, weights):
	kl_loss = - 0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
	l2_loss = mse(pred, gt).sum(1).sum(-1).sum(-1).mean()
	w_l2_loss = (l2_loss / weights).sum(-1).sum(-1).mean()
	l2_loss = l2_loss.sum(-1).sum(-1).mean()
	return l2_loss, w_l2_loss, kl_loss


def mdn_loss(mu, z, logvar):
	l2_loss = 0.5 * (mse(mu, z) / logvar.exp()).sum(-1).mean()
	return l2_loss


class DivColor(object):
	def __init__(self, device):
		self.vae = VAEMod()
		self.mdn = MDNMod()
		self.best_loss_vae = np.inf
		self.best_loss_mdn = np.inf
		self.transform_vae =
		self.transform_mdn =
		self.device = device
		return

	def load_model(self):
		return

	def train_vae(self, dataloader):
		self.vae.train()
		total_train_loss = 0
		l2_train_loss = 0
		w_l2_train_loss = 0
		kl_train_loss = 0
		for idx, batch in enumerate(dataloader):
			img_l, img_ab, img_weights = self.transform_vae(batch)  # int8->float32->normalize (01, -11, -11)->split l, ab, weights
			mu, logvar, pred = self.vae(img_ab, img_l)
			l2_loss, w_l2_loss, kl_loss = vae_loss(mu, logvar, pred, img_ab, img_weights)
			loss = l2_loss + w_l2_loss + kl_loss
			loss.backward()
			self.optimizer_vae.step()
			total_train_loss += loss.item()
			l2_train_loss += l2_loss.item()
			w_l2_train_loss += w_l2_loss.item()
			kl_train_loss += kl_loss.item()
		total_train_loss /= (idx + 1)
		l2_train_loss /= (idx + 1)
		w_l2_train_loss /= (idx + 1)
		kl_train_loss /= (idx + 1)
		return total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss

	def eval_vae(self, dataloader):
		self.vae.eval()
		total_eval_loss = 0
		l2_eval_loss = 0
		w_l2_eval_loss = 0
		kl_eval_loss = 0
		with torch.no_grad():
			for idx, batch in enumerate(dataloader):
				img_l, img_ab, img_weights = self.transform_vae(
					batch)  # int8->float32->normalize (01, -11, -11)->split l, ab, weights
				mu, logvar, pred = self.vae(img_ab, img_l)
				l2_loss, w_l2_loss, kl_loss = vae_loss(mu, logvar, pred, img_ab, img_weights)
				loss = l2_loss + w_l2_loss + kl_loss
				total_eval_loss += loss.item()
				l2_eval_loss += l2_loss.item()
				w_l2_eval_loss += w_l2_loss.item()
				kl_eval_loss += kl_loss.item()
		total_eval_loss /= (idx + 1)
		l2_eval_loss /= (idx + 1)
		w_l2_eval_loss /= (idx + 1)
		kl_eval_loss /= (idx + 1)
		return total_eval_loss, l2_eval_loss, w_l2_eval_loss, kl_eval_loss

	def train_mdn(self, dataloader):
		self.mdn.train()
		train_loss = 0
		for idx, batch in enumerate(dataloader):
			img_l, z, logvar = self.transform_mdn(
				batch)  # int8->float32->normalize 01->split l, z, logvar->device
			mu = self.mdn(img_l)
			loss = mdn_loss(mu, z, logvar)
			loss.backward()
			self.optimizer_mdn.step()
			train_loss += loss.item()
		train_loss /= (idx + 1)
		return train_loss

	def eval_mdn(self, dataloader):
		self.mdn.eval()
		eval_loss = 0
		with torch.no_grad():
			for idx, batch in enumerate(dataloader):
				img_l, z, logvar = self.transform_mdn(
					batch)  # int8->float32->normalize 01->split l, z, logvar->device
				mu = self.mdn(img_l)
				loss = mdn_loss(mu, z, logvar)
				eval_loss += loss.item()
		eval_loss /= (idx + 1)
		return eval_loss

	def predict_one_image(self, path_in, path_out=None):
		"""
		Colorize one image.
		:param path: str, image path
		:return: none
		"""
		self.mdn.eval()
		self.vae.eval()
		dpi = 400
		h, w = 64, 64
		img = skimage.io.imread(path_in, as_gray=True)  # h, w
		# normalize?
		h_org, w_org = img.shape
		img = skimage.transform.resize(img, (h, w))
		img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(-1).unsqueeze(-1)  # 1, 1, h, w
		with torch.no_grad():
			z = self.mdn(img)
			sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.vae.cond_encoder(img)
		color_out = self.vae.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4).squeeze()
		color_out = color_out.cpu().numpy()
		color_out = np.transpose(color_out, (1, 2, 0))
		# un normalize
		color_out = skimage.transform.resize(color_out, (h_org, w_org, 3))
		fig = plt.figure(frameon=False)
		fig.set_size_inches(w, h)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(color_out, aspect='normal')
		fig.savefig(path_out, dpi)
		return

	def predict_images(self, x):
		# TODO: write this function
		return

	def fit_vae(self, epochs, lr=2e-4, wd=0.):
		self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=wd)
		self.vae_train_loss = np.zeros((epochs, 4))
		self.vae_val_loss = np.zeros((epochs, 4))
		for epoch in tqdm(range(epochs)):
			total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss = self.train_vae()
			total_val_loss, l2_val_loss, w_l2_val_loss, kl_val_loss = self.eval_vae()
			if total_val_loss < self.best_loss_vae:
				torch.save(self.vae.state_dict(), "models/divcolor_vae.pth")
				self.best_loss_vae = total_val_loss
			self.vae_train_loss[epoch] = [total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss]
			self.vae_val_loss[epoch] = [total_val_loss, l2_val_loss, w_l2_val_loss, kl_val_loss]
		return

	def fit_mdn(self, epochs, lr=2e-4, wd=0.):
		self.optimizer_mdn = torch.optim.Adam(self.mdn.parameters(), lr=lr, weight_decay=wd)
		self.mdn_train_loss = np.zeros(epochs)
		self.mdn_val_loss = np.zeros(epochs)
		for epoch in tqdm(range(epochs)):
			train_loss = self.train_mdn()
			val_loss = self.eval_mdn()
			if val_loss < self.best_loss_mdn:
				torch.save(self.mdn.state_dict(), "models/divcolor_mdn.pth")
				self.best_loss_mdn = val_loss
			self.mdn_train_loss[epoch] = train_loss
			self.mdn_val_loss[epoch] = val_loss
		return

	def make_latent_space(self, val_ab):
		self.vae.eval()
		# normalize and transform, etc
		with torch.no_grad():
			mu, logvar = self.vae.encoder(val_ab)
		mu = mu.cpu().numpy()
		logvar = logvar.cpu().numpy()
		np.save("../datasets/stl10/train_latent", mu)
		np.save("../datasets/stl10/train_latent", logvar)
		return

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--e", type=int, default=2, help="epochs (default 2)")
parser.add_argument("--bs", type=int, default=20, help="batch size (default 20)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="load pretrained model  (default False)")
parser.add_argument("--nf", type=int, default=4, help="number of filters  (default 4)")
parser.add_argument("--nl", type=int, default=3, help="number of conv layers  (default 3)")
args = parser.parse_args()
device = args.d
print(args)
print(device)

seed_everything()

train_lab = torch.tensor(np.load("../datasets/stl10/train_lab_1.npy"))
test_lab = torch.tensor(np.load("../datasets/stl10/test_lab.npy"))
val_lab = torch.tensor(np.load("../datasets/stl10/val_lab_1.npy"))

transform = torchvision.transforms.Compose([ToType(torch.float, device), Normalize()])

train_lab_set = torch.utils.data.TensorDataset(train_lab)
test_lab_set = torch.utils.data.TensorDataset(test_lab)
val_lab_set = torch.utils.data.TensorDataset(val_lab)

trainloader = torch.utils.data.DataLoader(train_lab_set, batch_size=args.bs, shuffle=True)
testloader = torch.utils.data.DataLoader(test_lab_set, batch_size=args.bs, shuffle=True)
valloader = torch.utils.data.DataLoader(val_lab_set, batch_size=args.bs, shuffle=True)

# save model hyperparameters
out_ch = 2
in_ch = 1
nf = args.nf
nlayers = args.nl
ks = 3
names = ["out_ch", "in_ch", "nf", "nlayers", "ks"]
values = [out_ch, in_ch, nf, nlayers, ks]
save_hyperparamters(names, values, "model_dec")

# define model
# model = DEC(out_ch=out_ch, in_ch=in_ch, nf=args.nf, nlayers=args.nl, ks=ks)
model = AE(out_ch, in_ch, nf, ks=ks)
model.load_state_dict(torch.load("models/dec.pth", map_location=args.d)) if args.pre else 0
model.to(device)
print(count_parameters(model))

# define train conditions
wd = 0.
dpi = 400

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss(reduction="none").to(device)
mae = torch.nn.L1Loss(reduction="none").to(device)

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
	train_loss = train_my_model(model, optimizer, trainloader)
	test_loss = eval_my_model(model, testloader)
	losses[epoch] = [train_loss, test_loss]
	print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss, test_loss))
	if test_loss < best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/dec.pth")
		best_loss = test_loss
	np.save("files/losses_dec", losses)
