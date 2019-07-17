from tqdm import tqdm
import argparse
import pandas as pd

from utils import *
from divcolor import *


def eval_vae(vae, cl, cab):
    vae.eval()
    with torch.no_grad():
        mu, _, _ = vae(cab, cl)
    return mu


def train_my_model(model, vae, optimizer, dataloader):
    train_loss = 0
    model.train()
    for idx, (batch) in tqdm(enumerate(dataloader)):
        cl, cab = transform(batch[0])
        optimizer.zero_grad()
        z = model(cl)
        mu = eval_vae(vae, cl, cab)
        loss = mse(z, mu.detach()).sum(-1).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= (idx + 1)
    return train_loss

def eval_my_model(model, vae, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (batch) in tqdm(enumerate(dataloader)):
            cl, cab = transform(batch[0])
            z = model(cl)
            mu = eval_vae(vae, cl, cab)
            loss = mse(z, mu.detach()).sum(-1).mean()
            test_loss += loss.item()
    test_loss /= (idx + 1)
    return test_loss


parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--e", type=int, default=2, help="epochs (default 2)")
parser.add_argument("--bs", type=int, default=20, help="batch size (default 20)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--pre", action="store_true", help="load pretrained model  (default False)")
parser.add_argument("--nf", type=int, default=4, help="number of filters  (default 4)")
parser.add_argument("--hs", type=int, default=2, help="hidden size  (default 2)")
args = parser.parse_args()
device = args.d
print(args)
print(device)

seed_everything()

# train_lab = torch.tensor(np.load("../datasets/stl10/resized64/train_lab_1.npy"))
# test_lab = torch.tensor(np.load("../datasets/stl10/resized64/test_lab.npy"))
# val_lab = torch.tensor(np.load("../datasets/stl10/resized64/val_lab_1.npy"))

h = 64
w = 64
c = 3

train_lab = torch.randn((args.bs, c, h, w))
test_lab = torch.randn((args.bs, c, h, w))
val_lab = torch.randn((args.bs, c, h, w))

transform = torchvision.transforms.Compose([ToType(torch.float, device), Normalize()])

train_lab_set = torch.utils.data.TensorDataset(train_lab)
test_lab_set = torch.utils.data.TensorDataset(test_lab)
val_lab_set = torch.utils.data.TensorDataset(val_lab)

trainloader = torch.utils.data.DataLoader(train_lab_set, batch_size=args.bs, shuffle=True)
testloader = torch.utils.data.DataLoader(test_lab_set, batch_size=args.bs, shuffle=True)
valloader = torch.utils.data.DataLoader(val_lab_set, batch_size=args.bs, shuffle=True)

# load vae hyperparameters

params = pd.read_csv("vae_divcolor_params.csv")

nf_vae = int(params["nf"].values[0])
hs_vae = int(params["hs"].values[0])

vae = VAE(nf=nf_vae, hs=hs_vae)
vae.load_state_dict(torch.load("models/vae_divcolor.pth", map_location=args.d))
vae.to(device)

# save hyperparameters
df = {"nf": [args.nf], "hs": [args.hs]}
df = pd.DataFrame(data=df)
df.to_csv("mdn_divcolor_params.csv", index=False)

mdn = MDN(nf=args.nf, hs=args.hs)
mdn.load_state_dict(torch.load("models/mdn.pth", map_location=args.d)) if args.pre else 0
mdn.to(device)
print(count_parameters(mdn))

wd = 0.
dpi = 400

optimizer = torch.optim.Adam(mdn.parameters(), lr=args.lr, weight_decay=wd)
mse = torch.nn.MSELoss(reduction="none").to(device)

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
	train_loss = train_my_model(mdn, vae, optimizer, trainloader)
	test_loss = eval_my_model(mdn, vae, testloader)
	losses[epoch] = [train_loss, test_loss]
	print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss, test_loss))
	if test_loss < best_loss:
		print("Saving")
		torch.save(mdn.state_dict(), "models/vae_divcolor.pth")
		best_loss = test_loss
		np.save("losses_mdn_divcolor", losses)
np.save("losses_mdn_divcolor", losses)