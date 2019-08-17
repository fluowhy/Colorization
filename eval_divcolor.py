from main_divcolor import *
from main import *

divcolor = DivColor("cpu", True)

divcolor.load_model()

test_grey = np.load("../datasets/stl10/test_grey_64.npy")
print(test_grey.shape)
divcolor.colorize_images(test_grey[:30])
