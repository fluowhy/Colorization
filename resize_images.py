import skimage
import numpy as np
from tqdm import tqdm


def resize_image(img, h=64, w=64):
    img = skimage.transform.resize(img, (h, w, 3))
    return img


def resize_all_images(path, savepath):
    img_org = np.load(path)
    img_org = np.transpose(img_org, (0, 2, 3, 1))
    n = len(img_org)
    img_resize = np.zeros((n, 64, 64, 3), dtype=np.uint8)
    for i in tqdm(range(n)):
        img_resize[i] = resize_image(img_org[i]) * 255
    img_resize = np.transpose(img_resize, (0, 3, 1, 2))
    np.save(savepath, img_resize)
    return


if __name__ == "__main__":
    savepath = "../datasets/stl10/train_rgb_64"
    path = "../datasets/stl10/train_rgb.npy"
    resize_all_images(path, savepath)

    savepath = "../datasets/stl10/test_rgb_64"
    path = "../datasets/stl10/test.npy"
    resize_all_images(path, savepath)

    savepath = "../datasets/stl10/val_rgb_64"
    path = "../datasets/stl10/val_rgb.npy"
    resize_all_images(path, savepath)


