import numpy as np
import cv2
import os
from tqdm import tqdm


def rgb2npy(set, split):
    path = "../datasets/stl10/grey/{}".format(split)
    paths = os.listdir(path)
    n = len(paths) - 1
    images = np.zeros((n, 96, 96, 3), dtype=np.uint8)
    for i in tqdm(range(n)):
        img = cv2.imread("../datasets/stl10/rgb/{}/{}/img_{}.png".format(set, split, str(i)))
        images[i] = img
    images = np.transpose(images, (0, 3, 1, 2))
    np.save("{}_{}_rgb".format(split, set), images)
    return


if __name__ == "__main__":
    rgb2npy("mine", "train")
    rgb2npy("mine", "test")
    rgb2npy("other", "train")
    rgb2npy("other", "test")
    rgb2npy("original", "train")
    rgb2npy("original", "test")
