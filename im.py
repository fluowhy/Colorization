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
    return (x - b[0]) / (b[1] - b[0])


def normalize_lab(x):
    if len(x.shape) == 4:
        x_l = normalize_l(x[:, 0])
        x_a = normalize_a(x[:, 1])
        x_b = normalize_b(x[:, 2])
    else:
        x_l = normalize_l(x[0])
        x_a = normalize_a(x[1])
        x_b = normalize_b(x[2])
    return torch.cat((x_l, x_a, x_b))


def gaussian_gramm(xi, xj, bw=1, trace_norm=True):
    x = (-0.5 * (xi.unsqueeze(1) - xj.unsqueeze(0)).pow(2) / bw ** 2).exp()
    if trace_norm:
        return x/x.trace()
    else:
        return x


def alpha_entropy(x, alpha=2):
    eigval, eigvec = torch.eig(x)
    return torch.sum(eigval[:, 0].pow(alpha))/(1 - alpha)


def alpha_entropy_gen(x, y, alpha=2):
    return alpha_entropy(x*y/torch.trace(x*y), alpha)


def mi(x, y, bw=1):
    xg = gaussian_gramm(x, x, bw)
    yg = gaussian_gramm(y, y, bw)
    return alpha_entropy(xg) + alpha_entropy(yg) - alpha_entropy_gen(xg, yg)


if __name__ == "__main__":
    img = cv2.imread("C:/Users/mauricio/Pictures/test_image_1.jpg")
    img = uint8_2_float(img).transpose()
    img = torch.tensor(img).float()

    #img_r = torch.ones((1, 10, 10))*0
    #img_g = torch.ones((1, 10, 10))
    #img_b = torch.ones((1, 10, 10))*0
    #img = torch.cat((img_r, img_g, img_b), dim=0)
    """
    l_lim = [0., 100.]
    a_lim = [- 107.8573, 107.8573]
    b_lim = [- 86.18303, 80.09231]
    """
    img_lab = colors.rgb_to_lab(img)
    img_nor = normalize_lab(img_lab).numpy()
    img_lab = img_lab.numpy()

    N = 100
    vecA = torch.randn(N)
    vecB = torch.randn(N)
    A = gaussian_gramm(vecA, vecA, bw=2)
    B = gaussian_gramm(vecB, vecB, bw=2)
    SA = alpha_entropy(A)
    SB = alpha_entropy(B)
    SAB = alpha_entropy_gen(A, B)

    mi = mi(vecA, vecB, bw=2)

    print(np.max(img_lab[0]), np.min(img_lab[0]))
    print(np.max(img_lab[1]), np.min(img_lab[1]))
    print(np.max(img_lab[2]), np.min(img_lab[2]))

    print(np.max(img_nor[0]), np.min(img_nor[0]))
    print(np.max(img_nor[1]), np.min(img_nor[1]))
    print(np.max(img_nor[2]), np.min(img_nor[2]))
