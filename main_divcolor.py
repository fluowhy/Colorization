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
        self.transform_mdn = GreyTransform(torch.float, device)
        self.unnormalize = UnNormalize(device)
        self.device = device
        self.l1 = 1e-2
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
        for idx, batch in tqdm(enumerate(dataloader)):
            img_lab, img_weights = batch
            img_l, img_ab = self.transform_lab(img_lab)
            img_weights = img_weights.to(self.device)
            mu, logvar, pred = self.vae(img_ab, img_l)
            l2_loss, w_l2_loss, kl_loss = vae_loss(mu, logvar, pred, img_ab, img_weights)
            loss = l2_loss + w_l2_loss + kl_loss * self.l1
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
            for idx, batch in tqdm(enumerate(dataloader)):
                img_lab, img_weights = batch
                img_l, img_ab = self.transform_lab(img_lab)
                img_weights = img_weights.to(self.device)
                mu, logvar, pred = self.vae(img_ab, img_l)
                l2_loss, w_l2_loss, kl_loss = vae_loss(mu, logvar, pred, img_ab, img_weights)
                loss = l2_loss + w_l2_loss + kl_loss * self.l1
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
        for idx, batch in tqdm(enumerate(dataloader)):
            img_g, z, logvar = batch
            img_g = self.transform_mdn(img_g)
            z = z.to(self.device)
            logvar = logvar.to(self.device)
            mu = self.mdn(img_g)
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
            for idx, batch in tqdm(enumerate(dataloader)):
                img_g, z, logvar = batch
                img_g = self.transform_mdn(img_g)
                z = z.to(self.device)
                logvar = logvar.to(self.device)
                mu = self.mdn(img_g)
                loss = mdn_loss(mu, z, logvar)
                eval_loss += loss.item()
        eval_loss /= (idx + 1)
        return eval_loss

    def colorize_one_image(self, path_in, path_out):
        """
        Colorize one image.
        :param path: str, image path
        :return: none
        """
        self.load_model()
        self.mdn.eval()
        self.vae.eval()
        h, w = 64, 64
        img = skimage.io.imread(path_in, as_gray=True)  # h, w
        h_org, w_org = img.shape
        img = skimage.transform.resize(img, (h, w))
        img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0)  # 1, 1, h, w
        with torch.no_grad():
            z = self.mdn(img)
            sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.vae.cond_encoder(img)
            ab_out = self.vae.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
        ab_out = torch.cat((img, ab_out), dim=1)
        ab_out = self.unnormalize(ab_out).squeeze().cpu().numpy()
        ab_out = np.transpose(ab_out, (1, 2, 0)).astype(np.int8)
        color_out = skimage.color.lab2rgb(ab_out)
        color_out = skimage.transform.resize(color_out, (h_org, w_org, 3))
        color_out = (color_out * 255).astype(np.uint8)
        cv2.imwrite(path_out, cv2.cvtColor(color_out, cv2.COLOR_RGB2BGR))
        return

    def colorize_images(self, img):
        """
        Colorize a numpy array of images.
        :param x: uint8 numpy array (n, h, w)
        :return:
        """
        # TODO: write this function
        self.load_model()
        self.mdn.eval()
        self.vae.eval()
        n, _, _ = img.shape
        img = img.astype(np.float32) / 255
        img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(1)
        with torch.no_grad():
            z = self.mdn(img)
            sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.vae.cond_encoder(img)
            ab_out = self.vae.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
        ab_out = torch.cat((img, ab_out), dim=1)
        ab_out = self.unnormalize(ab_out).cpu().numpy()
        ab_out = np.transpose(ab_out, (0, 2, 3, 1)).astype(np.int8)
        for i in range(n):
            color_out = skimage.color.lab2rgb(ab_out[i])
            color_out = skimage.transform.resize(color_out, (96, 96, 3))
            color_out = (color_out * 255).astype(np.uint8)
            cv2.imwrite("../datasets/stl10/divcolor/{}.png".format(str(i)), cv2.cvtColor(color_out, cv2.COLOR_RGB2BGR))
        return

    def fit_vae(self, train_loader, val_loader, epochs=2, lr=2e-4, wd=0.):
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=wd)
        self.vae_train_loss = np.zeros((epochs, 4))
        self.vae_val_loss = np.zeros((epochs, 4))
        for epoch in range(epochs):
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
        for epoch in range(epochs):
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

    def make_latent_space(self, image_set, split):
        self.vae.load_state_dict(torch.load("models/divcolor_vae.pth", map_location=self.device))
        self.vae.eval()
        data_loader = torch.utils.data.DataLoader(image_set, batch_size=100, shuffle=False)
        n, _, _= image_set.tensors[1].shape
        latent_mu = np.empty((0, 64), dtype=np.float)
        latent_logvar = np.empty((0, 64), dtype=np.float)
        # normalize and transform, etc
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                img_lab, _ = batch
                img_l, img_ab = self.transform_lab(img_lab)
                mu, logvar = self.vae.encoder(img_ab)
                mu = mu.squeeze().cpu().numpy()
                logvar = logvar.squeeze().cpu().numpy()
                latent_mu = np.vstack((latent_mu, mu))
                latent_logvar = np.vstack((latent_logvar, logvar))
        np.save("../datasets/stl10/{}_latent_mu".format(split), latent_mu)
        np.save("../datasets/stl10/{}_latent_logvar".format(split), latent_logvar)
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

        self.train_mu = torch.tensor(np.load("../datasets/stl10/train_latent_mu.npy"), dtype=torch.float, device="cpu")
        self.val_mu = torch.tensor(np.load("../datasets/stl10/val_latent_mu.npy"), dtype=torch.float, device="cpu")

        self.train_logvar =  torch.tensor(np.load("../datasets/stl10/train_latent_logvar.npy"), dtype=torch.float, device="cpu")
        self.val_logvar = torch.tensor(np.load("../datasets/stl10/val_latent_logvar.npy"), dtype=torch.float, device="cpu")
        return

    def make_dataset(self):
        self.train_set = torch.utils.data.TensorDataset(self.train_grey, self.train_mu, self.train_logvar)
        self.val_set = torch.utils.data.TensorDataset(self.val_grey, self.val_mu, self.val_logvar)
        return

    def make_dataloader(self, bs):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=bs, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=bs, shuffle=True)
        return


if __name__ == "__main__":

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

    grey_dataset = GreyDataset()

    mse = torch.nn.MSELoss(reduction="none").to(device)

    divcolor = DivColor(args.d)

    divcolor.fit_vae(lab_dataset.train_loader, lab_dataset.val_loader, epochs=args.e, lr=args.lr)

    divcolor.make_latent_space(lab_dataset.train_set, "train")
    divcolor.make_latent_space(lab_dataset.val_set, "val")

    grey_dataset.load_data()
    grey_dataset.make_dataset()
    grey_dataset.make_dataloader(args.bs)

    divcolor.fit_mdn(grey_dataset.train_loader, grey_dataset.val_loader, epochs=args.e, lr=args.lr)

    # divcolor.colorize_one_image("C:/Users/mauricio/Pictures/IMG_20160710_212006.jpg", "C:/Users/mauricio/Pictures/grey.png")
    # divcolor.colorize_images(grey_dataset.train_grey.cpu().numpy())
