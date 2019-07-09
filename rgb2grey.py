from tqdm import tqdm
import cv2

from utils import *


def rgb2grey(split):
    dataset = torchvision.datasets.STL10(root="../datasets/stl10", split=split, download=False)
    labels = dataset.labels
    dataset = np.transpose(dataset.data, (0, 2, 3, 1))
    grey = skimage.color.rgb2grey(dataset) * 255
    grey = grey.astype(np.uint8)
    np.save("../datasets/stl10/grey/{}/targets".format(split), labels)
    for i in tqdm(range(grey.shape[0])):
        cv2.imwrite("../datasets/stl10/grey/{}/img_{}.png".format(split, str(i)), grey[i])
        rgb2bgr(dataset[i])
        cv2.imwrite("../datasets/stl10/rgb/original/{}/img_{}.png".format(split, str(i)), rgb2bgr(dataset[i]))
    return


if __name__ == "__main__":
    rgb2grey("train")
    rgb2grey("test")