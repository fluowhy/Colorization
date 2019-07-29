from tqdm import tqdm
import argparse

from utils import *
from decoder import *


def train_my_model(model, optimizer, dataloader):
	train_loss = 0
	model.train()
	for idx, (batch) in tqdm(enumerate(dataloader)):
		cl, cab = transform(batch[0])
		optimizer.zero_grad()
		ab_pred = model(cl)
		loss = mse(ab_pred, cab).sum(-1).sum(-1).sum(-1).mean()
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
			ab_pred = model(cl)
			loss = mse(ab_pred, cab).sum(-1).sum(-1).sum(-1).mean()
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
parser.add_argument("--nl", type=int, default=3, help="number of conv layers  (default 3)")
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

# save model hyperparameters
out_ch = 2
in_ch = 1
nf = args.nf
nlayers = args.nl
ks = 3
names = ["out_ch", "in_ch", "nf", "nlayers", "ks"]
values = [out_ch, in_ch, nf, nlayers, ks]
save_hyperparamters(names, values, "model_dec")

# define model
# model = DEC(out_ch=out_ch, in_ch=in_ch, nf=args.nf, nlayers=args.nl, ks=ks)
model = AE(out_ch, in_ch, nf, ks=ks)
model.load_state_dict(torch.load("models/dec.pth", map_location=args.d)) if args.pre else 0
model.to(device)
print(count_parameters(model))

# define train conditions
wd = 0.
dpi = 400

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
bce = torch.nn.BCELoss().to(device)
mse = torch.nn.MSELoss(reduction="none").to(device)
mae = torch.nn.L1Loss(reduction="none").to(device)

losses = np.zeros((args.e, 2))
best_loss = np.inf
for epoch in range(args.e):
	train_loss = train_my_model(model, optimizer, trainloader)
	test_loss = eval_my_model(model, testloader)
	losses[epoch] = [train_loss, test_loss]
	print("Epoch {} vae train loss {:.3f} test loss {:.3f}".format(epoch, train_loss, test_loss))
	if test_loss < best_loss:
		print("Saving")
		torch.save(model.state_dict(), "models/dec.pth")
		best_loss = test_loss
	np.save("files/losses_dec", losses)
