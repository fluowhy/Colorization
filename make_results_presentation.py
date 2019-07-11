import numpy as np
import cv2
from tqdm import tqdm

def make_image(n=10, sp=20):
    picture = np.ones((3 * 96 + 2 * sp, 96 * n, 3), dtype=np.uint8)
    for j in tqdm(range(n)):
        img_gt = cv2.imread("../datasets/stl10/rgb/original/test/img_{}.png".format(str(j)), 1)
        img_m = cv2.imread("../datasets/stl10/rgb/mine/test/img_{}.png".format(str(j)), 1)
        img_o = cv2.imread("../datasets/stl10/rgb/other/test/img_{}.png".format(str(j)), 1)
        picture[:96, 96 * j: 96 * (j + 1)] = img_gt
        picture[96 + sp:2*96 + sp, 96 * j: 96 * (j + 1)] = img_m
        picture[2 * 96 + sp * 2:, 96 * j: 96 * (j + 1)] = img_o
    cv2.imwrite("../datasets/stl10/results/img.png", cv2.resize(picture, (0, 0), fx=2., fy=2.))
    return


if __name__ == "__main__":
    make_image()