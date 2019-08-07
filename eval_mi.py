from main_mi import *


autoencoder = AutoEncoder("cpu", True)

val_grey = np.load("../datasets/stl10/val_grey_64.npy")

autoencoder.colorize_one_image("/home/mauricio/Pictures/cat.jpeg", "/home/mauricio/Pictures/cat_colorized.png")
autoencoder.colorize_one_image("/home/mauricio/Pictures/house.png", "/home/mauricio/Pictures/house_colorized.png")
autoencoder.colorize_images(val_grey[:10])