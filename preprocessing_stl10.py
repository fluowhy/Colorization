import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pdb

from utils import *


def compute_weights(image_numpy, binedges, lossweights):
    img_lossweights = np.zeros((image_numpy.shape[0], 2, image_numpy.shape[2], image_numpy.shape[3]))
    for i, img in tqdm(enumerate(image_numpy)):
        img_lossweights[i] = getweights(img[1:], binedges, lossweights)
    return img_lossweights


def compute_bin_wei():
    countbins = 1. / np.load("prior_probs.npy")
    binedges = np.load("ab_quantize.npy").reshape(2, 313)
    lossweights = {}
    for i in range(313):
        if binedges[0, i] not in lossweights:
            lossweights[binedges[0, i]] = {}
        lossweights[binedges[0, i]][binedges[1, i]] = countbins[i]
    return binedges, lossweights


def compute_image_weights(image_numpy, setname):
    # weigths for L1 loss
    binedges, lossweights = compute_bin_wei()

    # compute image weights
    img_lossweights = compute_weights(image_numpy, binedges, lossweights)

    # save arrays
    np.save("../datasets/cifar10/loss_weights_{}".format(setname), img_lossweights)
    return


def load_split_convert(train_tensor, test_tensor, unlabeled_tensor):
    # train val split

    indexes = np.arange(unlabeled_tensor.shape[0])

    train_idx, val_idx = train_test_split(indexes, test_size=8/100)

    new_train_tensor = torch.cat((train_tensor, unlabeled_tensor[train_idx]))

    print("train: {:.0f}".format(new_train_tensor.shape[0]))
    print("test: {:.0f}".format(test_tensor.shape[0]))
    print("val: {:.0f}".format(len(val_idx)))

    # save sets as (N, C, h, w)
    np.save("../datasets/stl10/train", new_train_tensor.numpy())
    np.save("../datasets/stl10/val", unlabeled_tensor[val_idx].numpy())
    np.save("../datasets/stl10/test", test_tensor.numpy())

    # convert to lab color space and save
    new_train_lab = np.zeros((new_train_tensor))
    train_lab = rgb2lab(new_train_tensor)
    val_lab = rgb2lab(unlabeled_tensor[val_idx])
    test_lab = rgb2lab(test_tensor)
    np.save("../datasets/stl10/train_lab", train_lab)
    np.save("../datasets/stl10/val_lab", val_lab)
    np.save("../datasets/stl10/test_lab", test_lab)
    return


def new_load_convert(debug, split="train"):
    N = 10
    dataset = torchvision.datasets.STL10(root="../datasets/stl10/{}".format(split), split=split, download=False)
    # save sets as (N, C, h, w)
    if not debug:
        np.save("../datasets/stl10/{}".format(split), dataset.data)
        np.save("../datasets/stl10/{}_lab".format(split), rgb2lab(torch.as_tensor(dataset.data)))
    else:
        np.save("../datasets/stl10/{}".format(split), dataset.data[:N])
        np.save("../datasets/stl10/{}_lab".format(split), rgb2lab(torch.as_tensor(dataset.data[:N])))
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument("--debug", action="store_true", help="select debugging state (default False)")

    args = parser.parse_args()

    new_load_convert(args.debug, split="train")
    new_load_convert(args.debug, split="test")
    new_load_convert(args.debug, split="unlabeled")


    # first split rgb
    train_data = np.load("../datasets/stl10/train.npy")
    unlabeled_data = np.load("../datasets/stl10/unlabeled.npy")

    indexes = np.arange(unlabeled_data.shape[0])

    train_idx, val_idx = train_test_split(indexes, test_size=8 / 100)

    np.save("../datasets/stl10/train_rgb", np.vstack((train_data, unlabeled_data[train_idx])))
    np.save("../datasets/stl10/val_rgb", unlabeled_data[val_idx])

    # now lab images
    train_data = np.load("../datasets/stl10/train_lab.npy")
    unlabeled_data = np.load("../datasets/stl10/unlabeled_lab.npy")

    np.save("../datasets/stl10/train_lab_1", np.vstack((train_data, unlabeled_data[train_idx])))
    np.save("../datasets/stl10/val_lab_1", unlabeled_data[val_idx])

    """
    compute_image_weights(train_lab.astype(np.float32), "train")
    compute_image_weights(val_lab.astype(np.float32), "val")
    compute_image_weights(test_lab.astype(np.float32), "test")
    """