from preprocessing import *
from constants import *

def show_img(image_name, img_npz_path=IMG_NPZ_DIR):
    if os.path.isfile("{}/{}.npz".format(img_npz_path, image_name)):
        img = np.load("{}/{}.npz".format(img_npz_path, image_name))["features"]
        img = img * 255
        img = np.array(img, dtype=np.uint8)
        Dataset.show(img)


if __name__ == '__main__':
    show_img('0000b55567e8c74_basic')
