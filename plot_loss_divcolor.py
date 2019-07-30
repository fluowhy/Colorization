import numpy as np
import matplotlib.pyplot as plt


def plot_loss_vae(train_loss, val_loss, dpi=400):
    plt.clf()
    plt.plot(train_loss[:, 1], color="navy", label="l2")
    plt.plot(train_loss[:, 2], color="navy", label="w_l2")
    plt.plot(train_loss[:, 3], color="navy", label="kl")
    plt.plot(val_loss[:, 1], color="red", label="l2")
    plt.plot(val_loss[:, 2], color="red", label="w_l2")
    plt.plot(val_loss[:, 3], color="red", label="kl")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mean loss value")
    # plt.ylim([0, 100])
    plt.savefig("figures/train_curve_divcolor_vae", dpi=dpi)
    return


if __name__ == "__main__":
    vae_train_loss = np.load("files/divcolor_vae_train_loss.npy")
    vae_val_loss = np.load("files/divcolor_vae_val_loss.npy")
    mdn_train_loss = np.load("files/divcolor_mdn_train_loss.npy")
    mdn_val_loss = np.load("files/divcolor_mdn_val_loss.npy")
    plot_loss_vae(vae_train_loss, vae_val_loss)
