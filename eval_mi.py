from main_mi import *

autoencoder = AutoEncoder("cpu", True)

test_grey = np.load("../datasets/stl10/test_grey_64.npy")
print(test_grey.shape)
autoencoder.colorize_images(test_grey[:30])