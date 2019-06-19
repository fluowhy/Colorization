import argparse
from tqdm import tqdm

from utils import *


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="colorization")
    parser.add_argument("--debug", action="store_true", help="select debugging state (default False)")
    parser.add_argument("--dataset", type=str, default="cifar10" help="select dataset cifar10 stl10 vocdet (default cifar10)")

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
    train_lab, test_lab = load_dataset(debug=args.debug, N=10, device=device, name=name)

    # compute image weights

    img_lossweights_train = np.zeros((train_lab.shape[0], 2, 32, 32))
    img_lossweights_test = np.zeros((test_lab.shape[0], 2, 32, 32))
    for i, img in tqdm(enumerate(train_lab)):
        img_lossweights_train[i] = getweights(img[1:].cpu().numpy(), binedges, lossweights)
    for i, img in tqdm(enumerate(test_lab)):
        img_lossweights_test[i] = getweights(img[1:].cpu().numpy(), binedges, lossweights)
    if args.debug:
        np.save("lossweights_train_{}_{}".format(name, "debug"), img_lossweights_train)
        np.save("lossweights_test_{}_{}".format(name, "debug"), img_lossweights_test)
    else:
        np.save("lossweights_train_{}".format(name), img_lossweights_train)
        np.save("lossweights_test_{}".format(name), img_lossweights_test)