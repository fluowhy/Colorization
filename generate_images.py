import argparse
from tqdm import tqdm
import cv2

from utils import *
from vae import *
from decoder import *


def generate_rgb_vae(model, split):
    model.eval()
    path = "../datasets/stl10/grey/{}".format(split)
    paths = os.listdir(path)    
    savepath = "../datasets/stl10/rgb/vae/{}".format(split)
    with torch.no_grad():
        for i in tqdm(range(len(paths) - 1)[:100]):
            img_name = "{}/img_{}.png".format(path, str(i))
            img = torch.tensor(cv2.imread(img_name, 0), dtype=torch.float, device=args.d)
            img = img.unsqueeze(0).unsqueeze(0) / 255
            mu_c, _, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = model.cond_encoder(img)
            ab_out = model.decoder(mu_c.unsqueeze(0), sc_feat32, sc_feat16, sc_feat8, sc_feat4)
            rgb_out = antitransform(torch.cat((img, ab_out), dim=1))
            rgb_out = np.transpose(rgb_out, (2, 3, 1, 0)).squeeze()
            cv2.imwrite("{}/img_{}.png".format(savepath, str(i)), rgb2bgr(rgb_out))
    return


def generate_rgb_vaegen(model, split):
    model.eval()
    path = "../datasets/stl10/grey/{}".format(split)
    paths = os.listdir(path)
    savepath = "../datasets/stl10/rgb/vaegen/{}".format(split)
    with torch.no_grad():
        for i in tqdm(range(len(paths) - 1)[:100]):
            img_name = "{}/img_{}.png".format(path, str(i))
            img = torch.tensor(cv2.imread(img_name, 0), dtype=torch.float, device=args.d)
            img = img.unsqueeze(0).unsqueeze(0) / 255
            mu, logvar, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = model.encode(img)
            stddev = torch.sqrt(torch.exp(logvar))
            sample = torch.randn(stddev.shape, device=stddev.device)
            z = torch.add(mu, torch.mul(sample, stddev))
            ab_out = model.decode(z.reshape((1, 16, 1, 1)), sc_feat32, sc_feat16, sc_feat8, sc_feat4)
            rgb_out = antitransform(torch.cat((img, ab_out), dim=1))
            rgb_out = np.transpose(rgb_out, (2, 3, 1, 0)).squeeze()
            cv2.imwrite("{}/img_{}.png".format(savepath, str(i)), rgb2bgr(rgb_out))
    return


def generate_rgb_dec(model, split):
    model.eval()
    path = "../datasets/stl10/grey/{}".format(split)
    paths = os.listdir(path)
    savepath = "../datasets/stl10/rgb/dec/{}".format(split)
    with torch.no_grad():
        for i in tqdm(range(len(paths) - 1)[:100]):
            img_name = "{}/img_{}.png".format(path, str(i))
            img = torch.tensor(cv2.imread(img_name, 0), dtype=torch.float, device=args.d)
            img = img.unsqueeze(0).unsqueeze(0) / 255
            ab_out = model(img)
            rgb_out = antitransform(torch.cat((img, ab_out), dim=1))
            rgb_out = np.transpose(rgb_out, (2, 3, 1, 0)).squeeze()
            cv2.imwrite("{}/img_{}.png".format(savepath, str(i)), rgb2bgr(rgb_out))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
    parser.add_argument("--model", type=str, default="vae", help="select model: vae, vaegen (default vae)")
    args = parser.parse_args()

    seed = 1111
    seed_everything(seed)

    device = args.d
    antitransform = torchvision.transforms.Compose([UnNormalize(), ToRGB()])

    if args.model == "vae":
        model = VAE96(in_ab=2, in_l=1, nf=32, ld=16, ks=3, do=0.7)
        model.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=args.d))
        generate_rgb_vae(model, "train")
        generate_rgb_vae(model, "test")
    elif args.model == "vaegen":        
        model = VAELAB(in_ab=2, in_l=1, nf=32, ld=16, ks=3, do=0.7)
        model.load_state_dict(torch.load("models/vae_lab.pth", map_location=args.d))
        generate_rgb_vaegen(model, "train")
        generate_rgb_vaegen(model, "test")
    elif args.model == "dec":
        model = DEC(out_ch=2, in_ch=1, nf=64, nlayers=5, ks=3)
        model.load_state_dict(torch.load("models/dec.pth", map_location=args.d))
        generate_rgb_dec(model, "train")
        generate_rgb_dec(model, "test")
        