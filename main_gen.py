import torch
import torchvision

import matplotlib.pyplot as plt

from tqdm import tqdm
import pytorch_colors as colors

from im import *
from model import *
from utils import *
from vae import *
from mdn import *


def vae_loss(mu, logvar, pred, gt):
    bs = gt.shape[0]
    kl_loss = - 0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    recon_loss_l2 = mse(pred.reshape((bs, -1)), gt.reshape((bs, -1)))
    return kl_loss, recon_loss_l2

seed_everything()

if not os.path.exists("figures"):
    os.makedirs("figures")
if not os.path.exists("models"):
    os.makedirs("models")

cuda = False
device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
print(device)

h, w = [32, 32]
N = 1000
bs = 100
lr = 2e-4
wd = 0.
epochs = 200
dpi = 400

train_lab, test_lab = load_dataset(N, device)

train_lab_set = torch.utils.data.TensorDataset(train_lab)
test_lab_set = torch.utils.data.TensorDataset(test_lab)

trainloader = torch.utils.data.DataLoader(train_lab_set, batch_size=bs, shuffle=True)
testloader = torch.utils.data.DataLoader(test_lab_set, batch_size=bs, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

vae = VAE(16, device).to(device)
discriminator = DIS(2, 16, 1, 3).to(device)
print(count_parameters(vae))
print(count_parameters(discriminator))
optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss().to(device)



losses = np.zeros((epochs, 4))
best_loss = np.inf
for epoch in range(epochs):
    vae.train()
    discriminator.train()
    train_loss_vae = 0
    train_loss_dis = 0
    k = 0
    for idx, (batch) in enumerate(trainloader):
        batch = batch[0]
        cL = batch[:, 0].unsqueeze(1)
        cab = batch[:, 1:]
        bs = cL.shape[0]
        optimizer.zero_grad()
        optimizer_dis.zero_grad()
        mu, logvar, color_out = vae(cab, cL)
        kl_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, cab)
        fake_labels = torch.zeros((bs, 1)).to(device)
        real_labels = torch.ones((bs, 1)).to(device)
        fake_pred = discriminator(color_out.detach())
        fake_pred_gen = discriminator(color_out)
        real_pred = discriminator(cab)
        loss_dis = 0.5*bce(1 - fake_pred, fake_labels) + 0.5*bce(real_pred, real_labels)
        loss_gen = bce(1 - fake_pred, fake_labels)
        loss_vae = (0.5*kl_loss + 0.5*recon_loss_l2)
        loss = loss_vae + bce(fake_pred_gen, fake_labels)
        loss.backward()
        loss_dis.backward()
        optimizer.step()
        optimizer_dis.step()
        train_loss_vae += loss.item()
        train_loss_dis += loss_dis.item()
        k += 1
    train_loss_vae /= (idx + 1)
    train_loss_dis /= (idx + 1)
    vae.eval()
    test_loss_vae = 0
    test_loss_dis = 0
    with torch.no_grad():
        for idx, (batch) in enumerate(testloader):
            batch = batch[0]
            cL = batch[:, 0].unsqueeze(1)
            cab = batch[:, 1:]
            mu, logvar, color_out = vae(cab, cL)
            kl_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, cab)
            fake_labels = torch.zeros((bs, 1)).to(device)
            real_labels = torch.ones((bs, 1)).to(device)
            fake_pred = discriminator(color_out.detach())
            real_pred = discriminator(cab)
            fake_pred_gen = discriminator(color_out)
            loss_dis = 0.5 * bce(1 - fake_pred, fake_labels) + 0.5 * bce(real_pred, real_labels)
            loss_vae = (0.5 * kl_loss + 0.5 * recon_loss_l2)
            loss = loss_vae + bce(fake_pred_gen, fake_labels)
            test_loss_vae += loss.item()
            test_loss_dis += loss_dis.item()
    test_loss_vae /= (idx + 1)
    test_loss_dis /= (idx + 1)
    print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss_vae + train_loss_dis, test_loss_vae + test_loss_dis))
    if test_loss_vae + test_loss_dis < best_loss:
        torch.save(vae.state_dict(), "models/gen.pth")
        best_loss = test_loss_vae + test_loss_dis
        print("saving")
    losses[epoch, :2] = [train_loss_vae, train_loss_dis]
    losses[epoch, 2:] = [test_loss_vae, test_loss_dis]


plt.clf()
plt.plot(losses[:, 0], color="navy", label="vae train")
plt.plot(losses[:, 1], color="blue", label="dis train")
plt.plot(losses[:, 2], color="red", label="vae test")
plt.plot(losses[:, 3], color="pink", label="dis test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("figures/train_curve", dpi=dpi)

vae.load_state_dict(torch.load("models/gen.pth"))
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