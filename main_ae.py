from tqdm import tqdm
import argparse

from model import *
from utils import *


def ae_loss(pred, gt, weights):
    l2_loss = mse(pred, gt).sum(-1).sum(-1).sum(-1).mean()
    return l2_loss


class AutoEncoder(object):
    def __init__(self, device, pre=False):
        self.device = device
        self.ae = AE()
        self.ae.to(device)
        self.load_model() if pre else 0
        self.best_loss_ae = np.inf
        self.transform_lab = torchvision.transforms.Compose([ToType(torch.float, device), Normalize(device)])
        self.transform_mdn = GreyTransform(torch.float, device)
        self.unnormalize = UnNormalize(device)
        self.reg1 = 1e-2
        self.reg2 = 0.5
        self.relu = torch.nn.ReLU()
        print(count_parameters(self.ae))

    def load_model(self):
        self.ae.load_state_dict(torch.load("models/ae.pth", map_location=self.device))
        return

    def train_ae(self, dataloader):
        self.ae.train()
        train_loss = 0
        for idx, batch in tqdm(enumerate(dataloader)):
            img_lab, img_weights = batch
            img_l, img_ab = self.transform_lab(img_lab)
            img_weights = img_weights.to(self.device)
            pred = self.ae(img_l)
            loss = ae_loss(pred, img_ab, img_weights)
            loss.backward()
            self.optimizer_ae.step()
            train_loss += loss.item()
        train_loss /= (idx + 1)
        return train_loss

    def eval_ae(self, dataloader):
        self.ae.eval()
        eval_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                img_lab, img_weights = batch
                img_l, img_ab = self.transform_lab(img_lab)
                img_weights = img_weights.to(self.device)
                pred = self.ae(img_l)
                loss = ae_loss(pred, img_ab, img_weights)
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
        self.ae.eval()
        h, w = 64, 64
        img = cv2.imread(path_in, 0)
        # resize image
        h_org, w_org = img.shape
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float) / 255
        img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0)  # 1, 1, h, w
        with torch.no_grad():
            ab_out = self.ae(img)
            ab_out = torch.clamp(ab_out, - 1., 1.)
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
        self.ae.eval()
        n, _, _ = img.shape
        img = img.astype(np.float32) / 255
        img = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(1)
        with torch.no_grad():
            ab_out = self.ae(img)
            ab_out = torch.clamp(ab_out, - 1., 1.)
        lab_out = torch.cat((img, ab_out), dim=1)
        lab_out = self.unnormalize(lab_out).cpu().numpy()
        lab_out = np.transpose(lab_out, (0, 2, 3, 1)).astype(np.uint8)
        for i in range(n):
            color_out = cv2.cvtColor(lab_out[i], cv2.COLOR_LAB2BGR)
            color_out = cv2.resize(color_out, (96, 96), interpolation=cv2.INTER_AREA)
            cv2.imwrite("../datasets/stl10/divcolor/ae_{}.png".format(str(i)), color_out)
        return

    def fit_ae(self, train_loader, val_loader, epochs=2, lr=2e-4, wd=0.):
        self.optimizer_ae = torch.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=wd)
        self.train_loss = []
        self.val_loss = []
        for epoch in range(epochs):
            train_loss = self.train_ae(train_loader)
            val_loss = self.eval_ae(val_loader)
            print("Epoch {} train loss {:.4f} val loss {:.4f}".format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss_ae:
                print("Saving ae")
                torch.save(self.ae.state_dict(), "models/ae.pth")
                self.best_loss_ae = val_loss
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            np.save("files/ae_train_loss", self.train_loss)
            np.save("files/ae_val_loss", self.val_loss)
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

    mse = torch.nn.MSELoss(reduction="none").to(device)

    autoencoder = AutoEncoder(args.d, args.pre)

    autoencoder.fit_ae(lab_dataset.train_loader, lab_dataset.val_loader, epochs=args.e, lr=args.lr, wd=0.)
