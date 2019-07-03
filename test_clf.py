import argparse
from tqdm import tqdm

from model import *
from utils import *

parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--ds", type=str, default="stl10", help="select dataset, options: stl10, stl10m, stl10b, (default stl10)")
args = parser.parse_args()

seed = 1111
seed_everything(seed)

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
    y_pred = 

