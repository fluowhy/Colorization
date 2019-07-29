import numpy as np
import matplotlib.pyplot as plt
import skimage
from tqdm import tqdm


def rgb2lab(img):
    """
    RGB to LAB image convertion.
    :param img: uint8 numpy array (h, w, c).
    :return:  float numpy array (h, w, c).
    """
    lab = skimage.color.rgb2lab(img)
    return lab


def convert_rgb_to_lab(images_rgb):
    """
    Convert a bunch of images from rgb to lab.
    :param images_rgb: uint8 numpy array (n, c, h, w)
    :return: int8 numpy array (n, c, h, w)
    """
    images_rgb = np.transpose(images_rgb, (0, 2, 3, 1))
    images_lab = np.zeros(images_rgb.shape, dtype=np.int8)
    for idx, image in tqdm(enumerate(images_rgb)):
        lab = rgb2lab(image)
        images_lab[idx] = lab
    images_lab = np.transpose(images_lab, (0, 3, 1, 2))
    return images_lab


if __name__ == "__main__":
    img_test = np.load("../datasets/stl10/test.npy")
    img_train = np.load("../datasets/stl10/train_rgb.npy")
    img_val = np.load("../datasets/stl10/val_rgb.npy")
    print(img_test.shape)
    images_lab = convert_rgb_to_lab(img_train)
    np.save("../datasets/stl10/train_lab", images_lab)
    images_lab = convert_rgb_to_lab(img_test)
    np.save("../datasets/stl10/test_lab", images_lab)
    images_lab = convert_rgb_to_lab(img_val)
    np.save("../datasets/stl10/val_lab", images_lab)

