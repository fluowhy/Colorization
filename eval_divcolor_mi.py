from main_divcolor_mi import *
from main import *

divcolor = DivColorMI("cpu", True)

divcolor.load_model()

test_grey = np.load("../datasets/stl10/test_grey_64.npy")
print(test_grey.shape)
divcolor.colorize_images(test_grey[:30])
