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
parser.add_argument("--ds", type=str, default="original", help="select dataset, options: original, mine, other, (default original)")
args = parser.parse_args()
print(args)

seed = 1111
seed_everything(seed)

device = args.d

transform = torchvision.transforms.Compose([ToType(torch.float, device)])

h, w = [96, 96]

# load data
x_train = np.load("train_{}_rgb.npy".format(args.ds))
x_test = np.load("test_{}_rgb.npy".format(args.ds))
y_train = np.load("../datasets/stl10/grey/train/targets.npy")
y_test = np.load("../datasets/stl10/grey/test/targets.npy")

# load train and val index
train_idx = np.load("train_idx.npy")
val_idx = np.load("val_idx.npy")

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
nclasses = len(y_test.unique())

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
