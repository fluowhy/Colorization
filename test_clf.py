import argparse
from sklearn.metrics import classification_report

from model import *
from utils import *

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--ds", type=str, default="original", help="select dataset, options: original, mine, other, (default original)")
args = parser.parse_args()
print(args)

seed = 1111
seed_everything(seed)
dpi = 500

device = args.d

test_images = np.load("test_{}_rgb.npy".format(args.ds))
test_labels = np.load("../datasets/stl10/grey/test/targets.npy")
test_labels = numpy2torch(test_labels, "cpu", torch.long)
test_images = numpy2torch(test_images, "cpu", torch.uint8)

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

nclasses = 10

clf = CONVCLF(inch=3, nch=2, nh=20, nout=nclasses).to(device)
clf.load_state_dict(torch.load("models/clf_{}.pth".format(args.ds), map_location=device))

clf.eval()
with torch.no_grad():
    y_pred = clf(test_images.to(device).float() / 255)
    y_pred = y_pred.argmax(1).cpu().numpy()
y_true = test_labels.squeeze().cpu().numpy()
plot_confusion_matrix(y_true, y_pred, np.arange(10), normalize=True, title=args.ds, cmap=plt.cm.Blues)

print(classification_report(y_true, y_pred))
