import numpy as np
import matplotlib.pyplot as plt


class PlotLoss(object):
    def __init__(self, readname, savename, limits=None):
        self.readname = readname
        self.savename = savename
        self.limits = limits

    def plot(self):
        plot_loss(self.readname, self.savename, self.limits)
        return


def plot_loss(path, savename, limits):
    dpi = 500
    loss = np.load(path)
    plt.clf()
    plt.plot(loss[:, 0], color="navy", label="train")
    plt.plot(loss[:, 1], color="red", label="test")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if limits != None:
        plt.ylim(limits)
    plt.savefig("figures/train_curve_{}".format(savename), dpi=dpi)
    return


if __name__ == "__main__":    
    dpi = 500
    losses = PlotLoss("files/losses.npy", "vae", [0, 10])
    losses_vae_lab = PlotLoss("files/losses_vae_lab.npy", "vaegen", [0, 10])
    losses_dec = PlotLoss("files/losses_dec.npy", "dec", [150, 175])
    losses_mdn_divcolor = PlotLoss("files/losses_mdn_divcolor.npy", "mdn_divcolor", [0, 0.002])
    losses_vae_divcolor = PlotLoss("files/losses_vae_divcolor.npy", "vae_divcolor", [0, 0.01])
    losses.plot()
    losses_vae_lab.plot()
    losses_dec.plot()
    losses_mdn_divcolor.plot()
    losses_vae_divcolor.plot()
