import os
from tqdm import tqdm


split = "train"
path = "../datasets/stl10/grey/{}".format(split)
paths = os.listdir(path)
for img_name in tqdm(paths[:10]):
    in_path = "{}/{}".format(path, img_name)
    out_path = "../datasets/stl10/rgb/other/{}/{}".format(split, img_name)
    print(in_path)
    print(out_path)
    script_path = "/home/mauricio/Documents/seminario/learnopencv/Colorization/colorizeImage.py"
    os.system("python {} --input {} --output {}".format(script_path, in_path, out_path))