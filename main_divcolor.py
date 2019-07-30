from tqdm import tqdm
import argparse

from divcolor import *
from utils import *


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
        self.vae.to(device)
        self.mdn.to(device)
        self.best_loss_vae = np.inf
        self.best_loss_mdn = np.inf
        self.transform_lab = torchvision.transforms.Compose([ToType(torch.float, device), Normalize(device)])
        self.transform_mdn = torchvision.transforms.Compose([ToType(torch.float, device), Normalize(device)])
        self.device = device
        print(count_parameters(self.vae))
        print(count_parameters(self.mdn))

    def load_model(self):
        self.vae.load_state_dict(torch.load("models/divcolor_vae.pth", map_location=self.device))
        self.mdn.load_state_dict(torch.load("models/divcolor_mdn.pth", map_location=self.device))
        return

    def train_vae(self, dataloader):
        self.vae.train()
        total_train_loss = 0
        l2_train_loss = 0
        w_l2_train_loss = 0
        kl_train_loss = 0
        for idx, batch in enumerate(dataloader):
            img_lab, img_weights = batch
            img_l, img_ab = self.transform_lab(img_lab)
            img_weights = img_weights.to(self.device)
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
                img_lab, img_weights = batch
                img_l, img_ab = self.transform_lab(img_lab)
                img_weights = img_weights.to(self.device)
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
        self.load_model()
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

    def fit_vae(self, train_loader, val_loader, epochs=2, lr=2e-4, wd=0.):
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=wd)
        self.vae_train_loss = np.zeros((epochs, 4))
        self.vae_val_loss = np.zeros((epochs, 4))
        for epoch in tqdm(range(epochs)):
            total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss = self.train_vae(train_loader)
            total_val_loss, l2_val_loss, w_l2_val_loss, kl_val_loss = self.eval_vae(val_loader)
            if total_val_loss < self.best_loss_vae:
                torch.save(self.vae.state_dict(), "models/divcolor_vae.pth")
                self.best_loss_vae = total_val_loss
            self.vae_train_loss[epoch] = [total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss]
            self.vae_val_loss[epoch] = [total_val_loss, l2_val_loss, w_l2_val_loss, kl_val_loss]
            np.save("files/divcolor_vae_train_loss", self.vae_train_loss)
            np.save("files/divcolor_vae_val_loss", self.vae_val_loss)
        return

    def fit_mdn(self, train_loader, val_loader, epochs=2, lr=2e-4, wd=0.):
        self.optimizer_mdn = torch.optim.Adam(self.mdn.parameters(), lr=lr, weight_decay=wd)
        self.mdn_train_loss = np.zeros(epochs)
        self.mdn_val_loss = np.zeros(epochs)
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_mdn(train_loader)
            val_loss = self.eval_mdn(val_loader)
            if val_loss < self.best_loss_mdn:
                torch.save(self.mdn.state_dict(), "models/divcolor_mdn.pth")
                self.best_loss_mdn = val_loss
            self.mdn_train_loss[epoch] = train_loss
            self.mdn_val_loss[epoch] = val_loss
            np.save("files/divcolor_mdn_train_loss", self.mdn_train_loss)
            np.save("files/divcolor_mdn_val_loss", self.mdn_val_loss)
        return

    def make_latent_space(self, train_set, val_set):
        # TODO: refactor
        self.vae.load_state_dict(torch.load("models/divcolor_vae.pth", map_location=self.device))
        self.vae.eval()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False)
        # normalize and transform, etc
        with torch.no_grad():
            mu, logvar = self.vae.encoder(val_ab)
        mu = mu.cpu().numpy()
        logvar = logvar.cpu().numpy()
        np.save("../datasets/stl10/train_latent_mu", mu)
        np.save("../datasets/stl10/train_latent_logvar", logvar)
        return


class LABDataset(object):
	def __init__(self):
		return

	def load_data(self):
		self.train_lab = torch.tensor(np.load("../datasets/stl10/train_lab_64.npy"), dtype=torch.int8, device="cpu")
		# self.test_lab = torch.tensor(np.load("../datasets/stl10/test_lab_64.npy"), dtype=torch.int8, device="cpu")
		self.val_lab = torch.tensor(np.load("../datasets/stl10/val_lab_64.npy"), dtype=torch.int8, device="cpu")

		self.train_hist = torch.tensor(np.load("../datasets/stl10/train_hist_values_64.npy"), dtype=torch.float, device="cpu")
		# self.test_hist = torch.tensor(np.load("../datasets/stl10/test_hist_values_64.npy"), dtype=torch.float, device="cpu")
		self.val_hist = torch.tensor(np.load("../datasets/stl10/val_hist_values_64.npy"), dtype=torch.float, device="cpu")

	def make_dataset(self):
		self.train_set = torch.utils.data.TensorDataset(self.train_lab, self.train_hist)
		# self.test_vae_set = torch.utils.data.TensorDataset(self.test_lab, self.test_hist)
		self.val_set = torch.utils.data.TensorDataset(self.val_lab, self.val_hist)
		return

	def make_dataloader(self, bs):
		self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=bs, shuffle=True)
		# self.test_loader = torch.utils.data.DataLoader(self.test_vae_set, batch_size=bs, shuffle=True)
		self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=bs, shuffle=True)
		return


class GreyDataset(object):
	def __init__(self):
		return

	def load_data(self):
		self.train_grey = torch.tensor(np.load("../datasets/stl10/train_grey_64.npy"), dtype=torch.uint8, device="cpu")
		self.val_grey = torch.tensor(np.load("../datasets/stl10/val_grey_64.npy"), dtype=torch.uint8, device="cpu")

		self.train_mu = torch.tensor(np.load("../datasets/stl10/train_latent_mu"), dtype=torch.float, device="cpu")
		self.val_mu = torch.tensor(np.load("../datasets/stl10/val_latent_mu"), dtype=torch.float, device="cpu")

		self.train_logvar =  torch.tensor(np.load("../datasets/stl10/train_latent_logvar"), dtype=torch.float, device="cpu")
		self.val_logvar = torch.tensor(np.load("../datasets/stl10/val_latent_logvar"), dtype=torch.float, device="cpu")
		return

	def make_dataset(self):
		self.train_set = torch.utils.data.TensorDataset(self.train_grey, self.train_mu, self.train_logvar)
		self.val_set = torch.utils.data.TensorDataset(self.val_grey, self.val_mu, self.val_logvar)
		return

	def make_dataloader(self, bs):
		self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=bs, shuffle=True)
		self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=bs, shuffle=True)
		return


seed_everything()

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--e", type=int, default=2, help="epochs (default 2)")
parser.add_argument("--bs", type=int, default=100, help="batch size (default 20)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="load pretrained model  (default False)")
args = parser.parse_args()
device = args.d
print(args)
print(device)

lab_dataset = LABDataset()
lab_dataset.load_data()
lab_dataset.make_dataset()
lab_dataset.make_dataloader(args.bs)

mse = torch.nn.MSELoss(reduction="none").to(device)

divcolor = DivColor(args.d)
divcolor.make_latent_space(lab_dataset.train_set, lab_dataset.val_set)
# divcolor.fit_vae(lab_dataset.train_loader, lab_dataset.val_loader, epochs=args.e, lr=args.lr)
