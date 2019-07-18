import skimage
import numpy as np
from tqdm import tqdm


def resize_image(img, h=64, w=64):
    img = skimage.transform.resize(img, (h, w, 3))
    return img


def resize_all_images(path, savepath):
    img_lab = np.load(path)
    img_lab = np.transpose(img_lab, (0, 2, 3, 1))
    n = len(img_lab)
    resized_lab = np.zeros((n, 64, 64, 3), dtype=np.int8)
    for i in tqdm(range(n)):
        resized_lab[i] = resize_image(img_lab[i])
    resized_lab = np.transpose(resized_lab, (0, 2, 3, 1))
    np.save(savepath, resized_lab)
    return


if __name__ == "__main__":
    savepath = "../datasets/stl10/resized64/train_lab_1.npy"
    path = "../datasets/stl10/train_lab_1.npy"
    resize_all_images(path, savepath)

    savepath = "../datasets/stl10/resized64/test_lab.npy"
    path = "../datasets/stl10/test_lab.npy"
    resize_all_images(path, savepath)

    savepath = "../datasets/stl10/resized64/val_lab_1.npy"
    path = "../datasets/stl10/val_lab_1.npy"
    resize_all_images(path, savepath)


