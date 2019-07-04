from tqdm import tqdm
import cv2

from vae import *
from utils import *


def make_images_and_file(images, split):
    file = open("list.{}.vae.txt".format(split), "w")
    for i in tqdm(range(images.shape[0])):
        path = "../images/{}/{}.png".format(split, str(i))
        file.write("{} \n".format(path))
        cv2.imwrite(path, np.transpose(images[i], (1, 2, 0)))
        cv2.destroyAllWindows()
    file.close()
    return


def make_features_and_file(images, split):
    file = open("list.{}.txt".format(split), "w")
    model = VAE96(in_ab=2, in_l=1, nf=64, ld=128, ks=3, do=0.7)  # 64, 128
    model.load_state_dict(torch.load("models/vae_mi_stl10.pth", map_location=device))
    model.to(device)
    transform = torchvision.transforms.Compose([ToLAB(device), Normalize()])
    images = torch.tensor(images, dtype=torch.uint8, device=device)
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    model.eval()
    j = 0
    with torch.no_grad():
        for idx, (batch) in tqdm(enumerate(dataloader)):
            img_l, _ = transform(batch[0])
            _, _, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = model.cond_encoder(img_l)
            sc_feat32 = sc_feat32.cpu().numpy()
            for i in range(img_l.shape[0]):
                path = "../feats/{}/{}".format(split, str(j))
                file.write("{}.npz \n".format(path))
                np.savez(path, sc_feat32[i].squeeze())
                j += 1
    file.close()
    return

dpi = 500
device = "cpu"
bs = 100

x_train = torchvision.datasets.STL10(root="../datasets/stl10", split="train", download=False).data
x_test = torchvision.datasets.STL10(root="../datasets/stl10", split="test", download=False).data

#make_images_and_file(x_train, "train")
#make_images_and_file(x_test, "test")

make_features_and_file(x_train, "train")
make_features_and_file(x_test, "test")

