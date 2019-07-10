import numpy as np
import matplotlib.pyplot as plt


def plot_loss(path):
    dpi = 500
    loss = np.load(path)
    plt.plot(loss[:, 0], color="navy", label="train")
    plt.plot(loss[:, 1], color="red", label="test")
    plt.grid(True)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("figures/train_curve", dpi=dpi)
    plt.show()
    return


if __name__ == "__main__":
    plot_loss("losses.npy")