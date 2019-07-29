from tqdm import tqdm
import argparse

from utils import *
from vae import *
from mdn import *


"""
entrenar con bs=187
"""

def vae_loss(mu, logvar, pred, gt):
    kl_loss = - 0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    recon_loss_l2 = mse(pred, gt).sum(-1).sum(-1).sum(-1).mean()
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
mse = torch.nn.MSELoss(reduction="none").to(device)

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
    vae.train()
    train_loss_vae = 0
    for idx, (batch) in tqdm(enumerate(trainloader)):
        cl, cab = transform(batch[0])
        optimizer.zero_grad()
        mu, logvar, color_out, mu_c, logvar_c = vae(cab, cl)
        kl_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, cab)
        l2_latent_space_mu = mse(mu_c, mu.detach()).sum(-1).mean()
        l2_latent_space_logvar = mse(logvar_c, logvar.detach()).sum(-1).mean()
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
            kl_loss, recon_loss_l2 = vae_loss(mu, logvar, color_out, cab)
            l2_latent_space_mu = mse(mu_c, mu.detach()).sum(-1).mean()
            l2_latent_space_logvar = mse(logvar_c, logvar.detach() ).sum(-1).mean()
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
    np.save("files/losses", losses)