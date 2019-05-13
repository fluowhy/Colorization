import torch
import pytorch_colors as colors
import numpy as np
import matplotlib.pyplot as plt
import cv2


def uint8_2_float(x):
    return x/255.


def float_2_uint8(x):
    return (x*255).astype(int)


def normalize_l(x, l=[0., 100.]):
    return (x - l[0])/(l[1] - l[0])


def normalize_a(x, a=[- 107.8573, 107.8573]):
    return (x - a[0])/(a[1] - a[0])


def normalize_b(x, b=[- 86.18303, 80.09231]):
    return (x - b[0])/(b[1] - b[0])


def normalize_lab(x):
    if len(x.shape)==4:
        x_l = normalize_l(x[:, 0])
        x_a = normalize_a(x[:, 1])
        x_b = normalize_b(x[:, 2])
    else:
        x_l = normalize_l(x[0])
        x_a = normalize_a(x[1])
        x_b = normalize_b(x[2])
    return torch.cat((x_l, x_a, x_b))

img = cv2.imread("C:/Users/mauricio/Pictures/test_image_1.jpg")

img = uint8_2_float(img).transpose()


#img_r = torch.ones((1, 10, 10))*0
#img_g = torch.ones((1, 10, 10))
#img_b = torch.ones((1, 10, 10))*0

l_lim = [0., 100.]
a_lim = [- 107.8573, 107.8573]
b_lim = [- 86.18303, 80.09231]

#img = torch.cat((img_r, img_g, img_b), dim=0)

img = torch.tensor(img).float()

img_lab = colors.rgb_to_lab(img)
img_nor = normalize_lab(img_lab).numpy()
img_lab = img_lab.numpy()
# cv2.imshow("image", (img_lab[0]*255/100).astype("uint8"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(np.max(img_lab[0]), np.min(img_lab[0]))
print(np.max(img_lab[1]), np.min(img_lab[1]))
print(np.max(img_lab[2]), np.min(img_lab[2]))

print(np.max(img_nor[0]), np.min(img_nor[0]))
print(np.max(img_nor[1]), np.min(img_nor[1]))
print(np.max(img_nor[2]), np.min(img_nor[2]))
