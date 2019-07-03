import torchvision
import numpy as np
from tqdm import tqdm
import cv2

from vae import *
from utils import *


def make_images(images, split):
    for i in tqdm(range(images.shape[0])):
        path = "/home/mauricio/Documents/seminario/images/{}/{}.png".format(split, str(i))
        cv2.imwrite(path, np.transpose(images[i], (1, 2, 0)))
        cv2.destroyAllWindows()
    return

dpi = 500
device = "cpu"
bs = 100

x_train = torchvision.datasets.STL10(root="../datasets/stl10", split="train", download=False).data
#x_test = torchvision.datasets.STL10(root="../datasets/stl10", split="test", download=False).data

#make_images(x_train, "train")
#make_images(x_test, "test")


model = VAE96(in_ab=2, in_l=1, nf=64, ld=128, ks=3, do=0.7)  # 64, 128
model.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=device))
model.to(device)
transform = torchvision.transforms.Compose([ToLAB(device), Normalize()])

x_train = torch.tensor(x_train, dtype=torch.uint8, device=device)
dataset = torch.utils.data.TensorDataset(x_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

model.eval()
with torch.no_grad():
    for idx, (batch) in enumerate(dataloader):
        img_l, _ = transform(batch[o])
        _, _, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = model.cond_encoder(img_l)
        f = 0
