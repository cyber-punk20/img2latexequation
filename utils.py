import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from constants import *
from preprocessing import Dataset, loadData, saveData


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


# def convert_imgs_npz(input_path, output_path):
#     for f in os.listdir(input_path):
#         if f.find(".png") != -1:
#             file_name = f[:f.find(".png")]
#             if os.path.exists("{}/{}.npz".format(output_path, file_name)):
#                 continue
#             img = Dataset.get_preprocessed_img("{}/{}".format(input_path, f), IMG_SIZE)
#             if img is None:
#                 continue
#             np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
#             retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

#             assert np.array_equal(img, retrieve)

#     print("Numpy arrays saved in {}".format(output_path))

def convert_imgs_npz(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for f in os.listdir(input_path):
        if f.endswith(".png"):
            file_name = f[:f.find(".png")]
            output_file = os.path.join(output_path, f"{file_name}.npz")
            if os.path.exists(output_file):
                continue
            img = Dataset.get_preprocessed_img(os.path.join(input_path, f), IMG_SIZE)
            if img is None:
                continue
            np.savez_compressed(output_file, features=img)
            retrieve = np.load(output_file)["features"]
            assert np.array_equal(img, retrieve)
    print("Numpy arrays saved in", output_path)

def check_img_height(input_path, img_height):
    import cv2
    for f in os.listdir(input_path):
        if f.endswith(".png"):
            file_name = f[:f.find(".png")]
            img_path = os.path.join(input_path, f)
            img = cv2.imread(img_path)
            # print(img_path)
            if img is None or img.size == 0:
                print(f"Failed to load image from path: {img_path}")
                continue
            (h, w) = img.shape[:2]
            if h != img_height:
                print(f"{img_path} with height {h} does not have height {img_height}")
                break

def check_img_width(input_path, img_width):
    import cv2
    for f in os.listdir(input_path):
        if f.endswith(".png"):
            file_name = f[:f.find(".png")]
            img_path = os.path.join(input_path, f)
            img = cv2.imread(img_path)
            # print(img_path)
            if img is None or img.size == 0:
                print(f"Failed to load image from path: {img_path}")
                continue
            (h, w) = img.shape[:2]
            if w != img_width:
                print(f"{img_path} with width {w} does not have width {img_width}")
                break

def convert_df(df, dataset):
    for index, row in df.iterrows():
        equ_token_id_seq = row['squashed_seq']
        equ_token_id_seq = [x for x in equ_token_id_seq if x != dataset.voc.vocabulary[END_TOKEN]]
        token_id_sequence = [dataset.voc.vocabulary[START_TOKEN]]
        token_id_sequence.extend(equ_token_id_seq)
        token_id_sequence.append(dataset.voc.vocabulary[END_TOKEN])
        # print(' '.join([dataset.voc.token_lookup[id] for id in token_id_sequence]))
        df.at[index, 'squashed_seq'] = token_id_sequence

if __name__ == "__main__":
    process_img = False
    process_df = False
    dataset = Dataset()
    dataset.voc.loadVolcabulary()
    dataset.voc.create_binary_representation()

    # check_img_width(IMG_DIR, 128)
    df_test = loadData('df_test.pkl')
    print(df_test)


    if process_img:
        print("convert_imgs_npz")
        convert_imgs_npz(IMG_DIR, IMG_NPZ_DIR)
    if process_df:
        
        df_test = loadData('df_test.pkl')
    
        convert_df(df_test, dataset)
        saveData('my_df_test.pkl', df_test)

        df_valid = loadData('df_valid.pkl')
        convert_df(df_valid, dataset)
        saveData('my_df_valid.pkl', df_valid)

        df_train = loadData('df_train.pkl')
        convert_df(df_train, dataset)
        saveData('my_df_train.pkl', df_train)

    # df = loadData('my_df_test.pkl')
    # seq = df.iloc[0]['squashed_seq']
    # print(' '.join([dataset.voc.token_lookup[id] for id in seq]))
    # df = loadData('df_test.pkl')
    # seq = df.iloc[0]['squashed_seq']
    # print(' '.join([dataset.voc.token_lookup[id] for id in seq]))

    # df = loadData('my_df_valid.pkl')
    # seq = df.iloc[0]['squashed_seq']
    # print(' '.join([dataset.voc.token_lookup[id] for id in seq]))
    # df = loadData('df_valid.pkl')
    # seq = df.iloc[0]['squashed_seq']
    # print(' '.join([dataset.voc.token_lookup[id] for id in seq]))

    # df = loadData('my_df_train.pkl')
    # seq = df.iloc[0]['squashed_seq']
    # print(' '.join([dataset.voc.token_lookup[id] for id in seq]))
    # df = loadData('df_train.pkl')
    # seq = df.iloc[0]['squashed_seq']
    # print(' '.join([dataset.voc.token_lookup[id] for id in seq]))
        