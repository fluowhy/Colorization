import argparse

from model import *
from utils import *

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--ds", type=str, default="stl10", help="select dataset, options: stl10, stl10m, stl10b, (default stl10)")
args = parser.parse_args()

seed = 1111
seed_everything(seed)
dpi = 500

if args.ds == "stl10":
    titlename = "stl10"
elif args.ds == "stl10b":
    titlename = "stl10_baseline"
elif args.ds == "stl10m":
    titlename = "stl10_model"

# loss plot

loss = np.load("loss_{}.npy".format(args.ds))

plt.clf()
plt.plot(loss[:, 0], color="navy", label="train")
plt.plot(loss[:, 1], color="red", label="test")
plt.plot(loss[:, 2], color="green", label="val")
plt.xlabel("epoch")
plt.ylabel("cross entropy")
plt.title("loss curves")
plt.legend()
plt.savefig("figures/loss_{}".format(args.ds), dpi=dpi)

device = args.d

h, w = [96, 96]

# testing only
nclasses = 10
N = 3 * nclasses

test_images = torch.randint(low=0, high=256, size=(N, 3, h, w), dtype=torch.uint8, device="cpu")
test_labels = torch.randint(low=0, high=nclasses, size=(N, 1), dtype=torch.long, device="cpu")

#classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
clf = CONVCLF(inch=3, nch=2, nh=20, nout=nclasses).to(device)
clf.load_state_dict(torch.load("models/clf_{}.pth".format(args.ds), map_location=device))

clf.eval()
with torch.no_grad():
    y_pred = clf(test_images.to(device).float() / 255)
    y_pred = y_pred.argmax(1)
plot_confusion_matrix(test_labels.squeeze().cpu().numpy(), y_pred.cpu().numpy(), np.arange(10), normalize=True, title=titlename, cmap=plt.cm.Blues)
