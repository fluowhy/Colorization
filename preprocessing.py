import argparse
from tqdm import tqdm

from utils import *


def compute_weights(set_lab, binedges, lossweights):
    img_lossweights = np.zeros((set_lab.shape[0], 2, set_lab.shape[2], set_lab.shape[3]))
    for i, img in tqdm(enumerate(set_lab)):
        img_lossweights[i] = getweights(img[1:].cpu().numpy(), binedges, lossweights)
    return img_lossweights


def save_array(array, name, debug):
    if debug:
        np.save("{}_{}".format(name, "debug"), array)
    else:
        np.save(name, array)
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument("--debug", action="store_true", help="select debugging state (default False)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="select dataset cifar10 stl10 vocdet (default "
                                                                       "cifar10)")

    device = "cpu"
    args = parser.parse_args()

    # weigths for L1 loss

    lossweights = None
    countbins = 1. / np.load("prior_probs.npy")
    binedges = np.load("ab_quantize.npy").reshape(2, 313)
    lossweights = {}
    for i in range(313):
        if binedges[0, i] not in lossweights:
            lossweights[binedges[0, i]] = {}
        lossweights[binedges[0, i]][binedges[1, i]] = countbins[i]

    # load dataset
    name = args.dataset
    train_lab, test_lab, unlabeled_lab = load_dataset(debug=args.debug, N=10, device=device, name=name)

    # compute image weights

    img_lossweights_train = compute_weights(train_lab, binedges, lossweights)
    img_lossweights_test = compute_weights(test_lab, binedges, lossweights)
    if name == "stl10":
        img_lossweights_unlabeled = compute_weights(unlabeled_lab, binedges, lossweights)
        save_array(img_lossweights_unlabeled, "lossweights_unlabeled_{}".format(name), args.debug)

    # save arrays

    save_array(img_lossweights_train, "lossweights_train_{}".format(name), args.debug)
    save_array(img_lossweights_test, "lossweights_test_{}".format(name), args.debug)


