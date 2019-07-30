from main_divcolor import *

grey_dataset = GreyDataset()

lab_dataset = LABDataset()
lab_dataset.load_data()
lab_dataset.make_dataset()

divcolor = DivColor("cpu")

divcolor.load_model()

val_grey = np.load("../datasets/stl10/val_grey_64.npy")

divcolor.colorize_one_image("/home/mauricio/Pictures/cat.jpeg", "/home/mauricio/Pictures/cat_grey.png")
divcolor.colorize_images(val_grey)
