import torch
#import pytorch_colors as colors
import numpy as np
import matplotlib.pyplot as plt
import cv2


def uint8_2_float(x):
    return x/255.


def float_2_uint8(x):
    return (x*255).astype(int)


def normalize(x, l):
    """

    Parameters
    ----------
    x : torch.tensor
        Images to be normalized.
    l : list
        Limits of the L color space.

    Returns
    -------
    torch.tensor
        Normalized L channel image.
    """
    return 2*(x - l[0])/(l[1] - l[0]) - 1


def unnormalize(x, l):
    """

    Parameters
    ----------
    x : torch.tensor
        Images to be unnormalized.
    l : list
        Limits of the L color space.

    Returns
    -------
    torch.tensor
        Unnormalized L channel image.
    """
    return 0.5*(l[1] - l[0])*(x + 1) + l[0]


def normalize_lab(x):
    """

    Parameters
    ----------
    x : torch.tensor
        Tensor batch shape (B, C, H, W) or (C, H, W).

    Returns
    -------
        Normalized Lab image from 0 to 1.
    """
    if len(x.shape) == 4:
        z = torch.zeros(x.shape)
        x_l = normalize(x[:, 0], [0., 100.])
        x_a = normalize(x[:, 1], [- 128., 127.])
        x_b = normalize(x[:, 2], [- 128., 127.])
        z[:, 0] = x_l
        z[:, 1] = x_a
        z[:, 2] = x_b
    else:
        z = torch.zeros(x.shape)
        x_l = normalize(x[0], [0., 100.])
        x_a = normalize(x[1], [- 128., 127.])
        x_b = normalize(x[2], [- 128., 127.])
        z[0] = x_l
        z[1] = x_a
        z[2] = x_b
    return z


def unnormalize_lab(x):
    """

    Parameters
    ----------
    x : torch.tensor
        Tensor batch shape (B, C, H, W) or (C, H, W).

    Returns
    -------
        Unormalized Lab image to original limits
    """
    if len(x.shape) == 4:
        z = torch.zeros(x.shape)
        x_l = unnormalize(x[:, 0], [0., 100.])
        x_a = unnormalize(x[:, 1], [- 128., 127.])
        x_b = unnormalize(x[:, 2], [- 128., 127.])
        z[:, 0] = x_l
        z[:, 1] = x_a
        z[:, 2] = x_b
    else:
        z = torch.zeros(x.shape)
        x_l = unnormalize(x[0], [0., 100.])
        x_a = unnormalize(x[1], [- 128., 127.])
        x_b = unnormalize(x[2], [- 128., 127.])
        z[0] = x_l
        z[1] = x_a
        z[2] = x_b
    return z


def gaussian_gramm(xi, xj, bw=1, trace_norm=True):
    x = (-0.5 * (xi.unsqueeze(1) - xj.unsqueeze(0)).pow(2).sum(dim=2) / bw ** 2).exp()
    if trace_norm:
        return x/x.trace()
    else:
        return x


def alpha_entropy(x, alpha=2):
    #eigval, eigvec = torch.eig(x)
    #return torch.sum(eigval[:, 0].pow(alpha))/(1 - alpha)
    return - torch.log(torch.trace(x.mm(x))/x.shape[0]**2)


def alpha_entropy_gen(x, y, alpha=2):
    xy = x*y
    return alpha_entropy(xy/torch.trace(xy), alpha)


def mi(x, y, bw=1):
    xg = gaussian_gramm(x, x, bw)
    yg = gaussian_gramm(y, y, bw)
    return alpha_entropy(xg) + alpha_entropy(yg) - alpha_entropy_gen(xg, yg)


if __name__ == "__main__":
    """
    img = cv2.imread("C:/Users/mauricio/Pictures/test_image_1.jpg")
    img = uint8_2_float(img)

    # img_r = torch.ones((1, 10, 10))*0
    # img_g = torch.ones((1, 10, 10))
    # img_b = torch.ones((1, 10, 10))*0
    # img = torch.cat((img_r, img_g, img_b), dim=0)

    plt.figure()
    plt.imshow(img)
    img = torch.tensor(np.transpose(img, (-1, 0, 1))).float()

    img_lab = colors.rgb_to_lab(img)
    img_nor = normalize_lab(img_lab)
    # lab to rgb
    img_lab_unor = unnormalize_lab(img_nor)
    img_rgb_from_lab = colors.lab_to_rgb(img_lab_unor.unsqueeze(0))
    plt.figure()
    plt.imshow(np.transpose(img_nor.numpy(), (1, 2, 0)))
    plt.figure()
    plt.imshow(np.transpose(img_rgb_from_lab.squeeze().numpy(), (1, 2, 0)))
    plt.show()
    """


    N = 10
    vecA = torch.randn((N, 2))
    vecB = torch.randn((N, 2))
    #A = gaussian_gramm(vecA, vecA, bw=2)
    #B = gaussian_gramm(vecB, vecB, bw=2)
    #SA = alpha_entropy(A)
    #SB = alpha_entropy(B)
    #SAB = alpha_entropy_gen(A, B)

    mi = mi(vecA, vecB, bw=2)