from tqdm import tqdm
import argparse

from utils import *
from divcolor import *


def vae_loss(mu, logvar, pred, gt):
    kl_loss = - 0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    recon_loss_l2 = mse(pred, gt).sum(-1).sum(-1).sum(-1).mean()
    return kl_loss, recon_loss_l2


def train_my_model(model, optimizer, dataloader):
    train_loss = 0
    model.train()
    for idx, (batch) in tqdm(enumerate(dataloader)):
        cl, cab = transform(batch[0])
        optimizer.zero_grad()
        mu, logvar, ab_pred = model(cab, cl)
        kl_loss, mse_loss = vae_loss(mu, logvar, ab_pred, cab)
        loss = (kl_loss + mse_loss) * 0.5
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= (idx + 1)
    return train_loss

def eval_my_model(model, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (batch) in tqdm(enumerate(dataloader)):
            cl, cab = transform(batch[0])
            mu, logvar, ab_pred = model(cab, cl)
            kl_loss, mse_loss = vae_loss(mu, logvar, ab_pred, cab)
            loss = (kl_loss + mse_loss) * 0.5
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

model = VAE(nf=args.nf, hs=args.hs)
model.load_state_dict(torch.load("models/vae_divcolor.pth", map_location=args.d)) if args.pre else 0
model.to(device)
print(count_parameters(model))

wd = 0.
dpi = 400

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
mse = torch.nn.MSELoss(reduction="none").to(device)

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
	train_loss = train_my_model(model, optimizer, trainloader)
	test_loss = eval_my_model(model, testloader)
	losses[epoch] = [train_loss, test_loss]
	print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss, test_loss))
	if test_loss < best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/vae_divcolor.pth")
		best_loss = test_loss
		np.save("losses_dec", losses)
np.save("losses_dec", losses)
