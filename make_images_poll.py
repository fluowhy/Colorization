import numpy as np
import cv2
from tqdm import tqdm

def make_image(comb, n=100, sp=20):
    orders = np.zeros((n, 2))
    for j in tqdm(range(n)):
        if comb == "gt_m":
            img_1 = cv2.imread("../datasets/stl10/rgb/original/test/img_{}.png".format(str(j)), 1)
            img_2 = cv2.imread("../datasets/stl10/rgb/mine/test/img_{}.png".format(str(j)), 1)
        elif comb == "gt_o":
            img_1 = cv2.imread("../datasets/stl10/rgb/original/test/img_{}.png".format(str(j)), 1)
            img_2 = cv2.imread("../datasets/stl10/rgb/other/test/img_{}.png".format(str(j)), 1)
        elif comb == "m_o":
            img_1 = cv2.imread("../datasets/stl10/rgb/other/test/img_{}.png".format(str(j)), 1)
            img_2 = cv2.imread("../datasets/stl10/rgb/mine/test/img_{}.png".format(str(j)), 1)
        images = [img_1, img_2]
        order = np.random.choice(np.arange(2, dtype=int), size=2, replace=False)
        picture = np.ones((96, 2  * 96 + 1 * sp, 3), dtype=np.uint8)
        for i in range(len(order)):
            if i == 1:
                picture[:, i * (96 + sp):] = images[order[i]]
            else:
                picture[:, i * (96 + sp):96 * (i + 1) + sp * i] = images[order[i]]
        cv2.imwrite("../datasets/stl10/poll/{}/img_{}.png".format(comb, str(j)), cv2.resize(picture, (0, 0), fx=2., fy=2.))
        orders[i] = order
    np.save("order_{}".format(comb), order)
    return


if __name__ == "__main__":
    combinations = ["gt_m", "gt_o", "m_o"]
    for c in combinations:
        order = make_image(c)

