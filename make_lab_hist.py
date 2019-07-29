import numpy as np
from tqdm import tqdm


def make_lab_hist_image_set(hist, images_lab):
    n, _, h, w = images_lab.shape
    images_hist_values = np.zeros((n, h, w))
    for i in tqdm(range(len(images_lab))):
        index_l = images_lab[i, 0].reshape(-1)
        index_a = images_lab[i, 1].reshape(-1)
        index_b = images_lab[i, 2].reshape(-1)
        hist_values = hist[index_l, index_a, index_b]
        images_hist_values[i] = hist_values.reshape(h, w)
    return images_hist_values


if __name__ == "__main__":
    train_lab = np.load("../datasets/stl10/train_lab_64.npy")
    val_lab = np.load("../datasets/stl10/val_lab_64.npy")
    test_lab = np.load("../datasets/stl10/test_lab_64.npy")

    concat = np.concatenate((train_lab, val_lab), axis=0)

    # -128, 127 to 0,255
    train_lab = train_lab + np.array([0, 128, 128]).reshape((1, 3, 1, 1))
    val_lab = val_lab + np.array([0, 128, 128]).reshape((1, 3, 1, 1))
    test_lab = test_lab + np.array([0, 128, 128]).reshape((1, 3, 1, 1))

    concat = concat + np.array([0, 128, 128]).reshape((1, 3, 1, 1))

    l = concat[:, 0].reshape(-1, 1)
    a = concat[:, 1].reshape(-1, 1)
    b = concat[:, 2].reshape(-1, 1)

    hist, edges = np.histogramdd(np.hstack((l, a, b)),
                                 density=True,
                                 bins=(np.arange(102) - 0.5, np.arange(256) - 0.5, np.arange(256) - 0.5))

    train_hist_values = make_lab_hist_image_set(hist, train_lab)
    val_hist_values = make_lab_hist_image_set(hist, val_lab)
    test_hist_values = make_lab_hist_image_set(hist, test_lab)

    np.save("../datasets/stl10/train_hist_values_64", train_hist_values)
    np.save("../datasets/stl10/val_hist_values_64", val_hist_values)
    np.save("../datasets/stl10/test_hist_values_64", test_hist_values)
