import torch
import torchvision

import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_colors as colors
import argparse
from sklearn.model_selection import train_test_split

from im import *
from model import *
from utils import *
from vae import *
from mdn import *


def vae_loss(mu, logvar, pred, gt, weights):
    bs = gt.shape[0]
    kl_loss = - 0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    recon_loss_l2 = mse(pred.reshape((bs, -1)), gt.reshape((bs, -1)))
    recon_loss = ((pred.reshape((bs, -1)) - gt.reshape((bs, -1))).pow(2) * weights.reshape((bs, -1))).sum(dim=1).mean()
    return kl_loss, recon_loss_l2, recon_loss


def loss_function(x_out, x, x_gray):
    bs = x.shape[0]
    mi_ch_a_g = mutual_info(x_out[:, 0].reshape(bs, -1), x_gray.reshape(bs, -1))
    mi_ch_b_g = mutual_info(x_out[:, 1].reshape(bs, -1), x_gray.reshape(bs, -1))
    return 1/mi_ch_a_g + 1/mi_ch_b_g


parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--e", type=int, default=2, help="epochs (default 2)")
parser.add_argument("--bs", type=int, default=20, help="batch size (default 20)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="load pretrained model  (default False)")

args = parser.parse_args()
device = args.d
print(device)

seed_everything()

make_folder()

h, w = [32, 32]
wd = 0.
dpi = 400

name = "stl10"

train_lab, test_lab, unlabeled_lab = load_dataset(debug=args.debug, N=10, device=device, name=name)

# load image weights

if args.debug:
    img_lossweights_train = np.load("lossweights_train_stl10_debug.npy")
    img_lossweights_test = np.load("lossweights_test_stl10_debug.npy")
    img_lossweights_unlabeled = np.load("lossweights_unlabeled_stl10_debug.npy")
    
else:
    img_lossweights_train = np.load("lossweights_train_stl10.npy")
    img_lossweights_test = np.load("lossweights_test_stl10.npy")
    img_lossweights_unlabeled = np.load("lossweights_unlabeled_stl10.npy")

# split unlabeled in train, test and val

N_train = img_lossweights_train.shape[0]
N_test = img_lossweights_test.shape[0]
N_unlabeled = img_lossweights_unlabeled.shape[0]
N_all = N_train + N_test + N_unlabeled
sum_train = int(0.7 * N_all - N_train)
sum_test = int(0.2 * N_all - N_test)
sum_val = int(0.1 * N_all - N_val)

indexes = np.arange(N_unlabeled)

train_idx, test_idx = train_test_split(indexes, test_size=sum_test/N_unlabeled)
train_idx, val_idx = train_test_split(train_idx, test_size=sum_val/len(train_idx))

print("train: {:.2f}".format((len(train_idx) + N_train) / N_all))
print("test: {:.2f}".format((len(test_idx) + N_test) / N_all))
print("val: {:.2f}".format((len(val_idx) + N_val) / N_all))

img_lossweights_train = torch.tensor(img_lossweights_train, dtype=torch.float, device=device)
img_lossweights_test = torch.tensor(img_lossweights_test, dtype=torch.float, device=device)

train_lab_set = torch.utils.data.TensorDataset(train_lab, img_lossweights_train)
test_lab_set = torch.utils.data.TensorDataset(test_lab, img_lossweights_test)

trainloader = torch.utils.data.DataLoader(train_lab_set, batch_size=args.bs, shuffle=True)
testloader = torch.utils.data.DataLoader(test_lab_set, batch_size=args.bs, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.debug:
    vae = VAE(2, device).to(device)
else:
    vae = VAE(16, device).to(device)
if args.pre:
    vae.load_state_dict(torch.load("models/vae_mi.pth"))
print(count_parameters(vae))
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss().to(device)
mutual_info = MutualInformation(2, 1.01, True, True).to(device)

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
    vae.train()
    train_loss_vae = 0
    for idx, (batch, img_weights) in tqdm(enumerate(trainloader)):
        cL = batch[:, 0].unsqueeze(1)
        cab = batch[:, 1:]
        optimizer.zero_grad()
        mu, logvar, color_out = vae(cab, cL)
        mi_loss = loss_function(color_out, cab, cL)
        kl_loss, recon_loss_l2, recon_loss = vae_loss(mu, logvar, color_out, cab, img_weights)
        loss_vae = (0.3*kl_loss + 0.3*recon_loss_l2 + 0.3*recon_loss + 0.1*mi_loss)
        loss_vae.backward()
        optimizer.step()
        train_loss_vae += loss_vae.item()
    train_loss_vae /= (idx + 1)
    vae.eval()
    test_loss_vae = 0
    with torch.no_grad():
        for idx, (batch, img_weights) in tqdm(enumerate(testloader)):
            cL = batch[:, 0].unsqueeze(1)
            cab = batch[:, 1:]
            mu, logvar, color_out = vae(cab, cL)
            mi_loss = loss_function(color_out, cab, cL)
            kl_loss, recon_loss_l2, recon_loss = vae_loss(mu, logvar, color_out, cab, img_weights)
            loss_vae = (0.3*kl_loss + 0.3*recon_loss_l2 + 0.3*recon_loss + 0.1*mi_loss)
            test_loss_vae += loss_vae.item()
    test_loss_vae /= (idx + 1)
    print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss_vae, test_loss_vae))
    if test_loss_vae < best_loss:
        torch.save(vae.state_dict(), "models/vae_mi.pth")
        best_loss = test_loss_vae
        print("saving")
    losses[epoch] = [train_loss_vae, test_loss_vae]


plt.clf()
plt.plot(losses[:, 0], color="navy", label="train")
plt.plot(losses[:, 1], color="red", label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("vae loss")
plt.savefig("figures/train_curve", dpi=dpi)

vae.load_state_dict(torch.load("models/vae_mi.pth"))
n = 10
l = 5
selected = np.random.choice(test_lab.shape[0], size=n, replace=False)
vae.eval()
img_lab = torch.zeros((n, 3, test_lab.shape[2], test_lab.shape[2]))
img_rgb = np.zeros((n, 3, test_lab.shape[2], test_lab.shape[2]))
img_gt_rgb = unnormalize_and_lab_2_rgb(test_lab[selected])
with torch.no_grad():
    _, _, ab = vae(test_lab[selected, 1:], test_lab[selected, 0].unsqueeze(1))
    img_lab[:, 1] = ab[:, 0]
    img_lab[:, 2] = ab[:, 1]
    img_lab[:, 0] = test_lab[selected, 0]
    img_rgb = unnormalize_and_lab_2_rgb(img_lab)
    for j in range(n):
        plt.clf()
        plt.axis("off")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2*l, l))
        ax[0].imshow(np.transpose(img_gt_rgb[j], (1, 2, 0)))
        ax[1].imshow(np.transpose(img_rgb[j], (1, 2, 0)))
        ax[0].set_title("ground truth")
        ax[1].set_title("colorized")
        plt.savefig("figures/im_{}".format(j), dpi=dpi)