import argparse
from tqdm import tqdm
import skimage

from utils import *
from vae import *


parser = argparse.ArgumentParser(description="colorization")
parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
parser.add_argument("--debug", action="store_true", help="select ot debugging state  (default False)")
parser.add_argument("--bs", type=int, default=100, help="batch size (default 100)")
args = parser.parse_args()

seed = 1111
seed_everything(seed)
dpi = 500

device = args.d

h, w = [96, 96]
transform = torchvision.transforms.Compose([ToLAB(device), Normalize()])
antitransform = torchvision.transforms.Compose([UnNormalize(), ToRGB()])

# load images
train_set = torchvision.datasets.STL10(root="../datasets/stl10", split="train", download=False)
test_set = torchvision.datasets.STL10(root="../datasets/stl10", split="test", download=False)

train_set = torch.utils.data.TensorDataset(torch.tensor(train_set.data, dtype=torch.uint8, device=device))
test_set = torch.utils.data.TensorDataset(torch.tensor(test_set.data, dtype=torch.uint8, device=device))

trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=False)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)

model = VAE96(in_ab=2, in_l=1, nf=64, ld=128, ks=3, do=0.7)  # 64, 128
model.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=args.d))


def generate_rgb(model, data_loader, split, imgshape):
    model.eval()
    new_rgb = np.zeros(imgshape)
    last_index = 0
    with torch.no_grad():
        for idx, (batch) in tqdm(enumerate(data_loader)):
            batch_l, _ = transform(batch[0])
            n = batch_l.shape[0]
            mu_c, _, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = model.cond_encoder(batch_l)
            ab_out = model.decoder(mu_c, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
            rgb_out = antitransform(torch.cat((batch_l, ab_out), dim=1))
            new_rgb[last_index:last_index + n] = rgb_out
            last_index += n
    new_rgb = np.array(new_rgb).squeeze()
    np.save("rgb_{}_{}".format(split, "vae"), new_rgb)
    return

generate_rgb(model, trainloader, "train", train_set.tensors[0].shape)
generate_rgb(model, testloader, "test", test_set.tensors[0].shape)