import argparse
from tqdm import tqdm

from model import *
from utils import *

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--e", type=int, default=2, help="epochs (default 2)")
parser.add_argument("--bs", type=int, default=20, help="batch size (default 20)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
parser.add_argument("--ds", type=str, default="stl10", help="select dataset, options: stl10, stl10m, stl10b, (default stl10)")
args = parser.parse_args()

seed = 1111
seed_everything(seed)

device = args.d

transform = torchvision.transforms.Compose([ToType(torch.float, device)])

h, w = [96, 96]
"""
# testing only
nclasses = 10
N = 3 * nclasses

train_images = torch.randint(low=0, high=256, size=(N, 3, h, w), dtype=torch.uint8, device="cpu")
test_images = torch.randint(low=0, high=256, size=(N, 3, h, w), dtype=torch.uint8, device="cpu")
val_images = torch.randint(low=0, high=256, size=(N, 3, h, w), dtype=torch.uint8, device="cpu")

train_labels = torch.randint(low=0, high=nclasses, size=(N, 1), dtype=torch.long, device="cpu")
test_labels = torch.randint(low=0, high=nclasses, size=(N, 1), dtype=torch.long, device="cpu")
val_labels = torch.randint(low=0, high=nclasses, size=(N, 1), dtype=torch.long, device="cpu")

train_set = torch.utils.data.TensorDataset(train_images, train_labels)
test_set = torch.utils.data.TensorDataset(test_images, test_labels)
val_set = torch.utils.data.TensorDataset(val_images, val_labels)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=True)
"""

# load data
x_train = torchvision.datasets.STL10(root="../datasets/stl10", split="train", download=False)
x_test = torchvision.datasets.STL10(root="../datasets/stl10", split="test", download=False)
y_train = x_train.labels
y_test = x_test.labels

# select dataset to use
if args.ds == "stl10":
    x_train = x_train.data
    x_test = x_test.data
if args.ds == "stl10m":
    x_train = np.load("rgb_train_vae.npy")
    x_test = np.load("rgb_test_vae.npy")
elif args.ds == "stl10b":
    f = 0

# load train and val index
train_idx = np.load("train_idx.npy")
val_idx = np.load("val_idx.npy")


def numpy2torch(x, device, dtype):
    return torch.tensor(x, device=device, dtype=dtype)

# convert to tensors
x_train = numpy2torch(x_train, "cpu", torch.uint8)
x_test = numpy2torch(x_test, "cpu", torch.uint8)
y_train = numpy2torch(y_train, "cpu", torch.long)
y_test = numpy2torch(y_test, "cpu", torch.long)

# split train val
x_val = x_train[val_idx]
x_train = x_train[train_idx]
y_val = y_train[val_idx]
y_train = y_train[train_idx]

# make dataset
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

# make dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

wd = 0.

clf = CONVCLF(inch=3, nch=2, nh=20, nout=nclasses).to(device)
print("classifier parameters {}".format(count_parameters(clf)))
optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=wd)
ce = torch.nn.CrossEntropyLoss(reduction="mean")


def train_my_model(model, optimizer, dataloader):
    model.train()
    train_loss = 0
    for idx, (batch, labels) in enumerate(dataloader):
        # transform
        optimizer.zero_grad()
        batch = transform(batch) / 255
        y_pred = model(batch)
        loss = ce(y_pred, labels.squeeze())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / (idx + 1)


def eval_my_model(model, dataloader):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for idx, (batch, labels) in enumerate(dataloader):
            batch = transform(batch) / 255
            y_pred = model(batch)
            loss = ce(y_pred, labels.squeeze())
            eval_loss += loss.item()
    return eval_loss / (idx + 1)


losses = np.zeros((args.e, 3))
best_loss = np.inf
for epoch in tqdm(range(args.e)):
    train_loss = train_my_model(clf, optimizer, train_loader)
    test_loss = eval_my_model(clf, test_loader)
    val_loss = eval_my_model(clf, val_loader)
    print("Epoch {} clf train loss {:.3f} test loss {:.3f} val loss {:.3f}".format(epoch, train_loss, test_loss, val_loss))
    if val_loss < best_loss:
        print("saving")
        torch.save(clf.state_dict(), "models/clf_{}.pth".format(args.ds))
        np.save("loss_{}".format(args.ds), losses)
        best_loss = val_loss
    losses[epoch] = [train_loss, test_loss, val_loss]
np.save("loss_{}".format(args.ds), losses)
