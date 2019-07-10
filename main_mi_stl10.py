import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from im import *
from model import *
from utils import *
from vae import *
from mdn import *


"""
entrenar con bs=187
"""

def vae_loss(mu, logvar, pred, gt):
    bs = gt.shape[0]
    kl_loss = - 0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    recon_loss_l2 = mse(pred.reshape((bs, -1)), gt.reshape((bs, -1))).mean()
    return kl_loss, recon_loss_l2


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
parser.add_argument("--nf", type=int, default=1, help="number of filters  (default 1)")
parser.add_argument("--ld", type=int, default=2, help="size of latent space  (default 2)")

args = parser.parse_args()
device = args.d
print(args)
print(device)

seed_everything()

make_folder()

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

vae = VAE96(in_ab=2, in_l=1, nf=args.nf, ld=args.ld, ks=3, do=0.7)  # 64, 128
vae.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=args.d)) if args.pre else 0
vae.to(device)
wd = 0.
dpi = 400
h, w = val_lab.shape[2], val_lab.shape[3]

print(count_parameters(vae))
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss(reduction="sum").to(device)
mae = torch.nn.L1Loss(reduction="sum")
# mutual_info = MutualInformation(2, 1.01, True, True).to(device)
# lam = 1

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
    vae.train()
    train_loss_vae = 0
    for idx, (batch) in tqdm(enumerate(trainloader)):
        cl, cab = transform(batch[0])
        optimizer.zero_grad()
        mu, logvar, color_out, mu_c, logvar_c = vae(cab, cl)
        #mi_loss = loss_function(color_out, cab, cl)
        kl_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, cab)
        l2_latent_space_mu = mse(mu_c, mu.detach()).mean()
        l2_latent_space_logvar = mse(logvar_c, logvar.detach()).mean()
        loss_vae = kl_loss + recon_loss_l2 + l2_latent_space_mu + l2_latent_space_logvar
        loss_vae.backward()
        optimizer.step()
        train_loss_vae += loss_vae.item()
    train_loss_vae /= (idx + 1)
    vae.eval()
    test_loss_vae = 0
    with torch.no_grad():
        for idx, (batch) in tqdm(enumerate(testloader)):
            cl, cab = transform(batch[0])
            mu, logvar, color_out, mu_c, logvar_c = vae(cab, cl)
            #mi_loss = loss_function(color_out, cab, cl)
            kl_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, cab)
            l2_latent_space_mu = mse(mu_c, mu.detach()).mean()
            l2_latent_space_logvar = mse(logvar_c, logvar.detach()).mean()
            loss_vae = kl_loss + recon_loss_l2 + l2_latent_space_mu + l2_latent_space_logvar
            test_loss_vae += loss_vae.item()
    test_loss_vae /= (idx + 1)
    print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss_vae, test_loss_vae))
    if test_loss_vae < best_loss:
        print("saving")
        torch.save(vae.state_dict(), "models/vae_mi_stl10.pth")
        best_loss = test_loss_vae
        np.save("losses", losses)
    losses[epoch] = [train_loss_vae, test_loss_vae]
np.save("losses", losses)

plt.clf()
plt.plot(losses[1:, 0], color="navy", label="train")
plt.plot(losses[1:, 1], color="red", label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("vae loss")
plt.savefig("figures/train_curve", dpi=dpi)

UN = UnNormalize()

vae.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=args.d))
n = 10
l = 5
selected = np.random.choice(test_lab.shape[0], size=n, replace=False)
vae.eval()
img_lab = torch.zeros((n, 3, h, w), dtype=torch.float, device=device)
img_gt_rgb = np.load("../datasets/stl10/test.npy")[selected]
with torch.no_grad():
    cl, _ = transform(test_lab[selected])
    ############
    mu_c, logvar_c, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = vae.cond_encoder(cl)
    stddev = torch.sqrt(torch.exp(logvar_c))
    sample = torch.randn(stddev.shape, device=stddev.device)
    z = torch.add(mu_c, torch.mul(sample, stddev))
    ab = vae.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
    ############
    img_lab[:, 1:] = ab
    img_lab[:, 0] = cl.squeeze()
    img_lab = UN(img_lab)
    img_rgb = lab2rgb(img_lab)
    nrows = int(n / 2)
    ncols = 4
    plt.clf()
    fig, row = plt.subplots(nrows=nrows, ncols=ncols, figsize=(l, l))
    for i, row in enumerate(row):
        row[0].imshow(np.transpose(img_gt_rgb[i], (1, 2, 0)))
        row[1].imshow(np.transpose(img_rgb[i], (1, 2, 0)))
        row[2].imshow(np.transpose(img_gt_rgb[i + nrows], (1, 2, 0)))
        row[3].imshow(np.transpose(img_rgb[i + nrows], (1, 2, 0)))
        row[0].set_title("ground truth") if i == 0 else 0
        row[1].set_title("colorized") if i == 0 else 0
        row[2].set_title("ground truth") if i == 0 else 0
        row[3].set_title("colorized") if i == 0 else 0
        row[0].axis("off")
        row[1].axis("off")
        row[2].axis("off")
        row[3].axis("off")
    plt.tight_layout()
    plt.savefig("figures/images", dpi=dpi)