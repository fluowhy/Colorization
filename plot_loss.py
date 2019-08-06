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
	train_loss = np.load("files/ae_train_loss.npy")
	val_loss = np.load("files/ae_val_loss.npy")
	plt.clf()
	plt.plot(train_loss, color="navy", label="train")
	plt.plot(val_loss, color="red", label="val")
	plt.savefig("figures/ae_loss", dpi=dpi)
