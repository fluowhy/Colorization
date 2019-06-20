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

    new_train_tensor = torch.cat((train_tensor.cpu(), unlabeled_tensor[train_idx].cpu()))

    print("train: {:.0f}".format(new_train_tensor.shape[0]))
    print("test: {:.0f}".format(test_tensor.shape[0]))
    print("val: {:.0f}".format(len(val_idx)))

    # save sets as (N, C, h, w)
    np.save("../datasets/stl10/train", new_train_tensor.cpu().numpy())
    np.save("../datasets/stl10/val", unlabeled_tensor[val_idx].cpu().numpy())
    np.save("../datasets/stl10/test", test_tensor.cpu().numpy())

    # convert to lab color space and save
    train_lab = rgb2lab(new_train_tensor.cpu())
    val_lab = rgb2lab(unlabeled_tensor[val_idx].cpu())
    test_lab = rgb2lab(test_tensor.cpu())
    np.save("../datasets/stl10/train_lab", train_lab)
    np.save("../datasets/stl10/val_lab", val_lab)
    np.save("../datasets/stl10/test_lab", test_lab)
    return train_lab, val_lab, test_lab


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument("--debug", action="store_true", help="select debugging state (default False)")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu")

    args = parser.parse_args()
    N = 10

    trainset = torchvision.datasets.STL10(root="../datasets/stl10/train", split="train", download=False)
    testset = torchvision.datasets.STL10(root="../datasets/stl10/test", split="test", download=False)
    unlabeledset = torchvision.datasets.STL10(root="../datasets/stl10/unlabeled", split="unlabeled", download=False)
    if not args.debug:
        train_tensor = torch.tensor(trainset.data, device=args.d)
        test_tensor = torch.tensor(testset.data, device=args.d)
        unlabeled_tensor = torch.tensor(unlabeledset.data, device=args.d)
    else:
        train_tensor = torch.tensor(trainset.data[:N], device=args.d)
        test_tensor = torch.tensor(testset.data[:N], device=args.d)
        unlabeled_tensor = torch.tensor(unlabeledset.data[:N], device=args.d)

    train_lab, val_lab, test_lab = load_split_convert(train_tensor, test_tensor, unlabeled_tensor)
    """
    compute_image_weights(train_lab.astype(np.float32), "train")
    compute_image_weights(val_lab.astype(np.float32), "val")
    compute_image_weights(test_lab.astype(np.float32), "test")
    """