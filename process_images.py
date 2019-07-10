import argparse
from tqdm import tqdm
import skimage
import cv2

from utils import *
from vae import *


def generate_rgb(model, split):
    model.eval()
    path = "../datasets/stl10/grey/{}".format(split)
    paths = os.listdir(path)
    with torch.no_grad():
        for i in tqdm(range(len(paths) - 1)):
            img_name = "{}/img_{}.png".format(path, str(i))
            img = torch.tensor(cv2.imread(img_name, 0), dtype=torch.float, device=args.d)
            img = img.unsqueeze(0).unsqueeze(0) / 255
            mu_c, _, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = model.cond_encoder(img)
            ab_out = model.decoder(mu_c.unsqueeze(0), sc_feat32, sc_feat16, sc_feat8, sc_feat4)
            rgb_out = antitransform(torch.cat((img, ab_out), dim=1))
            rgb_out = np.transpose(rgb_out, (2, 3, 1, 0)).squeeze()
            cv2.imwrite("../datasets/stl10/rgb/mine/{}/img_{}.png".format(split, str(i)), rgb2bgr(rgb_out))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
    args = parser.parse_args()

    seed = 1111
    seed_everything(seed)
    dpi = 500

    device = args.d

    model = VAE96(in_ab=2, in_l=1, nf=32, ld=16, ks=3, do=0.7)  # 64, 128
    model.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=args.d))

    antitransform = torchvision.transforms.Compose([UnNormalize(), ToRGB()])

    generate_rgb(model, "train")
    generate_rgb(model, "test")