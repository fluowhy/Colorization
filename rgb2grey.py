from tqdm import tqdm

from utils import *


def rgb2grey(split):
    rgb_images = np.load("../datasets/stl10/{}_rgb_64.npy".format(split))
    n, _, h, w = rgb_images.shape
    rgb_images = np.transpose(rgb_images, (0, 2, 3, 1))
    grey_images = np.zeros((n, h, w), dtype=np.uint8)
    for i in tqdm(range(len(rgb_images))):
        grey = skimage.color.rgb2grey(rgb_images[i]) * 255
        grey_images[i] = grey
    np.save("../datasets/stl10/{}_grey_64.npy".format(split), grey_images)
    return


if __name__ == "__main__":
    rgb2grey("train")
    rgb2grey("val")
    rgb2grey("test")