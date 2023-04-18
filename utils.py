import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from constants import *
from preprocessing import Dataset
def check_png_size(path):
    """
    Check if all PNG files under the directory have the same size.
    :param path: path to the directory
    :return: True if all PNG files have the same size, False otherwise
    """
    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
    if not png_files:
        print("No PNG files found under", path)
        return False
    sizes = set()
    for f in png_files:
        try:
            img = Image.open(os.path.join(path, f))
            sizes.add(img.size)
            img.close()
        except UnidentifiedImageError:
            print(f"Skipping {f} as it cannot be identified as an image file")
            continue

    if len(sizes) > 1:
        print("PNG files under", path, "have different sizes:")
        for s in sizes:
            print(s)
        return False

    print("All PNG files under", path, "have the same size:", sizes.pop())
    return True


def convert_imgs_npz(input_path, output_path):
    for f in os.listdir(input_path):
        if f.find(".png") != -1:
            file_name = f[:f.find(".png")]
            if os.path.exists("{}/{}.npz".format(output_path, file_name)):
                continue
            img = Dataset.get_preprocessed_img("{}/{}".format(input_path, f), IMG_SIZE)
            if img is None:
                continue

            np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
            retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

            assert np.array_equal(img, retrieve)

    print("Numpy arrays saved in {}".format(output_path))

if __name__ == "main":
    convert_imgs_npz(IMG_DIR, IMG_NPZ_DIR)