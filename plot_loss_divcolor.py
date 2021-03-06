import numpy as np
import matplotlib.pyplot as plt


def plot_loss_vae(train_loss, val_loss, dpi=400):
    l = 5

    plt.figure(figsize=(3 * l, l))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss[:, 1], color="navy", label="l2 train", linestyle="-")
    plt.plot(val_loss[:, 1], color="red", label="l2 val", linestyle="-")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mean loss value")
    plt.ylim([0, 2000])

    plt.subplot(1, 3, 2)
    plt.plot(train_loss[:, 2], color="navy", label="w_l2 train", linestyle="--")
    plt.plot(val_loss[:, 2], color="red", label="w_l2 val", linestyle="--")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mean loss value")
    plt.ylim([0, 2000])

    plt.subplot(1, 3, 3)
    plt.plot(train_loss[:, 3], color="navy", label="kl train", linestyle=":")
    plt.plot(val_loss[:, 3], color="red", label="kl val", linestyle=":")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mean loss value")
    plt.savefig("figures/train_curve_divcolor_vae", dpi=dpi)

    return


def plot_loss_mdn(train_loss, val_loss, dpi=400):
    l = 5

    plt.figure(figsize=(l, l))
    plt.plot(train_loss, color="navy", label="l2 train")
    plt.plot(val_loss, color="red", label="l2 val")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mean loss value")
    plt.ylim([0, 1e4])
    plt.savefig("figures/train_curve_divcolor_mdn", dpi=dpi)
    return


if __name__ == "__main__":
    vae_train_loss = np.load("files/divcolor_vae_train_loss.npy")
    vae_val_loss = np.load("files/divcolor_vae_val_loss.npy")
    mdn_train_loss = np.load("files/divcolor_mdn_train_loss.npy")
    mdn_val_loss = np.load("files/divcolor_mdn_val_loss.npy")
    plot_loss_vae(vae_train_loss, vae_val_loss)
    plot_loss_mdn(mdn_train_loss, mdn_val_loss)
