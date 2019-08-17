from tqdm import tqdm
import argparse

from divcolor import *
from utils import *


def mi_loss(pred, latent):
    eps = 1e-10
    mi_a = mutual_information(pred[:, 0].view(-1, 64 * 64), latent.view(-1, 2048))
    mi_b = mutual_information(pred[:, 1].view(-1, 64 * 64), latent.view(-1, 2048))
    mi = 1 / (mi_a + eps) + 1 / (mi_b + eps)
    return mi


def vae_loss(mu, logvar, pred, gt, weights):
    kl_loss = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
    l2_loss = mse(pred, gt).sum(1) # .sum(-1).sum(-1).mean()
    w_l2_loss = (l2_loss / weights).sum(-1).sum(-1).mean()
    l2_loss = l2_loss.sum(-1).sum(-1).mean()
    return l2_loss, w_l2_loss, kl_loss


def mdn_loss(mu, z, logvar):
    eps = 1e-10
    l2_loss = 0.5 * (mse(mu, z) / (logvar.exp() + eps)).sum(-1).mean()
    return l2_loss


class DivColorMI(object):
    def __init__(self, device, pre=False):
        self.device = device
        self.vae = VAEMod()
        self.mdn = MDNMod()
        self.vae.to(device)
        self.mdn.to(device)
        self.load_model() if pre else 0
        self.best_loss_vae = np.inf
        self.best_loss_mdn = np.inf
        self.transform_lab = torchvision.transforms.Compose([ToType(torch.float, device), Normalize(device)])
        self.transform_mdn = GreyTransform(torch.float, device)
        self.unnormalize = UnNormalize(device)
        self.reg1 = 1e-2
        self.reg2 = 1e-1
        print(count_parameters(self.vae))
        print(count_parameters(self.mdn))

    def load_model(self):
        self.vae.load_state_dict(torch.load("models/divcolor_mi_vae.pth", map_location=self.device))
        self.mdn.load_state_dict(torch.load("models/divcolor_mi_mdn.pth", map_location=self.device))
        return

    def train_vae(self, dataloader):
        self.vae.train()
        total_train_loss = 0
        l2_train_loss = 0
        w_l2_train_loss = 0
        kl_train_loss = 0
        mi_train_loss = 0
        for idx, batch in tqdm(enumerate(dataloader)):
            img_lab, img_weights = batch
            img_l, img_ab = self.transform_lab(img_lab)
            img_weights = img_weights.to(self.device)
            mu, logvar, pred = self.vae(img_ab)
            l2_loss, w_l2_loss, kl_loss = vae_loss(mu, logvar, pred, img_ab, img_weights)
            mi_loss = mi_loss(img_ab, mu)
            loss = w_l2_loss + kl_loss * self.reg1 + mi_loss * self.reg2
            loss.backward()
            self.optimizer_vae.step()
            total_train_loss += loss.item()
            l2_train_loss += l2_loss.item()
            w_l2_train_loss += w_l2_loss.item()
            kl_train_loss += kl_loss.item()
            mi_train_loss += mi_loss.item()
        total_train_loss /= (idx + 1)
        l2_train_loss /= (idx + 1)
        w_l2_train_loss /= (idx + 1)
        kl_train_loss /= (idx + 1)
        mi_train_loss /= (idx + 1)
        return total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss, mi_train_loss

    def eval_vae(self, dataloader):
        self.vae.eval()
        total_eval_loss = 0
        l2_eval_loss = 0
        w_l2_eval_loss = 0
        kl_eval_loss = 0
        mi_train_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                img_lab, img_weights = batch
                img_l, img_ab = self.transform_lab(img_lab)
                img_weights = img_weights.to(self.device)
                mu, logvar, pred = self.vae(img_ab)
                l2_loss, w_l2_loss, kl_loss = vae_loss(mu, logvar, pred, img_ab, img_weights)
                mi_loss = mi_loss(img_ab, mu)
                loss = w_l2_loss + kl_loss * self.reg1
                total_eval_loss += loss.item()
                l2_eval_loss += l2_loss.item()
                w_l2_eval_loss += w_l2_loss.item()
                kl_eval_loss += kl_loss.item()
                mi_train_loss += mi_loss.item()
        total_eval_loss /= (idx + 1)
        l2_eval_loss /= (idx + 1)
        w_l2_eval_loss /= (idx + 1)
        kl_eval_loss /= (idx + 1)
        mi_train_loss /= (idx + 1)
        return total_eval_loss, l2_eval_loss, w_l2_eval_loss, kl_eval_loss, mi_train_loss

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
        img = cv2.imread(path_in, 0)
        # resize image
        h_org, w_org = img.shape
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float) / 255
        img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0)  # 1, 1, h, w
        with torch.no_grad():
            z = self.mdn(img)
            ab_out = self.vae.decode(z)
        lab_out = torch.cat((img, ab_out), dim=1)
        lab_out = self.unnormalize(lab_out).squeeze().cpu().numpy()
        lab_out = np.transpose(lab_out, (1, 2, 0)).astype(np.uint8)
        color_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        color_out = cv2.resize(color_out, (w_org, h_org), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path_out, color_out)
        return

    def colorize_images(self, img):
        """
        Colorize a numpy array of images.
        :param x: uint8 numpy array (n, h, w)
        :return:
        """
        self.load_model()
        self.mdn.eval()
        self.vae.eval()
        n, _, _ = img.shape
        img = img.astype(np.float32) / 255
        img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(1)
        with torch.no_grad():
            z = self.mdn(img)
            ab_out = self.vae.decode(z)
        lab_out = torch.cat((img, ab_out), dim=1)
        lab_out = self.unnormalize(lab_out).cpu().numpy()
        lab_out = np.transpose(lab_out, (0, 2, 3, 1)).astype(np.uint8)
        for i in range(n):
            color_out = cv2.cvtColor(lab_out[i], cv2.COLOR_LAB2BGR)
            color_out = cv2.resize(color_out, (96, 96), interpolation=cv2.INTER_AREA)
            cv2.imwrite("../datasets/stl10/divcolor/mi_{}.png".format(str(i)), color_out)
        return

    def fit_vae(self, train_loader, val_loader, epochs=2, lr=2e-4, wd=0.):
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=wd)
        self.vae_train_loss = []
        self.vae_val_loss = []
        for epoch in range(epochs):
            total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss, mi_train_loss = self.train_vae(train_loader)
            total_val_loss, l2_val_loss, w_l2_val_loss, kl_val_loss, mi_val_loss = self.eval_vae(val_loader)
            print("Epoch {} train loss {:.4f} val loss {:.4f}".format(epoch, total_train_loss, total_val_loss))
            if total_val_loss < self.best_loss_vae:
                torch.save(self.vae.state_dict(), "models/divcolor_mi_vae.pth")
                self.best_loss_vae = total_val_loss
                print("Saving vae")
            self.vae_train_loss.append([total_train_loss, l2_train_loss, w_l2_train_loss, kl_train_loss, mi_train_loss])
            self.vae_val_loss.append([total_val_loss, l2_val_loss, w_l2_val_loss, kl_val_loss, mi_val_loss])
            np.save("files/divcolor_mi_vae_train_loss", self.vae_train_loss)
            np.save("files/divcolor_mi_vae_val_loss", self.vae_val_loss)
        return

    def fit_mdn(self, train_loader, val_loader, epochs=2, lr=2e-4, wd=0.):
        self.optimizer_mdn = torch.optim.Adam(self.mdn.parameters(), lr=lr, weight_decay=wd)
        self.mdn_train_loss = []
        self.mdn_val_loss = []
        for epoch in range(epochs):
            train_loss = self.train_mdn(train_loader)
            val_loss = self.eval_mdn(val_loader)
            print("Epoch {} train loss {:.4f} val loss {:.4f}".format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss_mdn:
                torch.save(self.mdn.state_dict(), "models/divcolor_mi_mdn.pth")
                self.best_loss_mdn = val_loss
                print("Saving mdn")
            self.mdn_train_loss.append(train_loss)
            self.mdn_val_loss.append(val_loss)
            np.save("files/divcolor_mi_mdn_train_loss", self.mdn_train_loss)
            np.save("files/divcolor_mi_mdn_val_loss", self.mdn_val_loss)
        return

    def make_latent_space(self, image_set, split):
        self.vae.load_state_dict(torch.load("models/divcolor_mi_vae.pth", map_location=self.device))
        self.vae.eval()
        data_loader = torch.utils.data.DataLoader(image_set, batch_size=100, shuffle=False)
        n, _, _= image_set.tensors[1].shape
        latent_mu = np.empty((0, 64), dtype=np.float)
        latent_logvar = np.empty((0, 64), dtype=np.float)
        print("generating latent space")
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                img_lab, _ = batch
                img_l, img_ab = self.transform_lab(img_lab)
                mu, logvar = self.vae.encode(img_ab)
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
    parser.add_argument("--lr_vae", type=float, default=2e-4, help="learning rate (default 2e-4)")
    parser.add_argument("--lr_mdn", type=float, default=2e-4, help="learning rate (default 2e-4)")
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

    divcolor = DivColor(args.d, args.pre)

    divcolor.fit_vae(lab_dataset.train_loader, lab_dataset.val_loader, epochs=args.e, lr=args.lr_vae, wd=0.)

    divcolor.make_latent_space(lab_dataset.train_set, "train")
    divcolor.make_latent_space(lab_dataset.val_set, "val")

    grey_dataset.load_data()
    grey_dataset.make_dataset()
    grey_dataset.make_dataloader(args.bs)

    divcolor.fit_mdn(grey_dataset.train_loader, grey_dataset.val_loader, epochs=args.e, lr=args.lr_mdn, wd=0.)
