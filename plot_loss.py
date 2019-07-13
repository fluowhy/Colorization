import numpy as np
import matplotlib.pyplot as plt


def plot_loss(path, savename):
    dpi = 500
    loss = np.load(path)
    plt.clf()
    plt.plot(loss[:, 0], color="navy", label="train")
    plt.plot(loss[:, 1], color="red", label="test")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim([0, 300])
    plt.savefig("figures/train_curve_{}".format(savename), dpi=dpi)
    return


if __name__ == "__main__":    
    dpi = 500
    plot_loss("losses.npy", "vae")
    plot_loss("losses_vae_lab.npy", "vaegen")
    plot_loss("losses_dec.npy", "dec")
