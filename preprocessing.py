import json
import pandas as pd
import numpy as np
import os
import sys

from constants import *


PLACEHOLDER = " "
START_TOKEN = '\\bos'
END_TOKEN = '\\eos'

def loadData(filename: str)-> pd.DataFrame:
    df = pd.read_pickle(os.path.join(DATA_DIR, filename))
    # print(df['seq_len'].equals(df['squashed_len']))
    return df.sort_index()

class Vocabulary:
    def __init__(self):
        self._dir = VOCAB_DIR
        # self._df_word2id = pd.read_pickle(os.path.join(self._dir, WORD2ID_FILENAME))
        self.token_lookup = pd.read_pickle(os.path.join(self._dir, ID2WORD_FILENAME))
        
        self.vocabulary = {} # word2id
        self.binary_vocabulary = {}

        self.size = 0

        # self.loadVolcabulary()
    def loadVolcabulary(self):
        for ind in self.token_lookup:
            key = self.token_lookup[ind]
            self.vocabulary[key] = ind
        self.size = len(self.vocabulary)
        self.append(PLACEHOLDER)
        

    def create_binary_representation(self):
        if sys.version_info >= (3,):
            items = self.vocabulary.items()
        else:
            items = self.vocabulary.iteritems()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary
    
    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1


class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences_ids = []
        self.next_word_ids = []

        self.voc = Vocabulary()
    
    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        # print(img_path)
        if img is None or img.size == 0:
            print(f"Failed to load image from path: {img_path}")
            return None
        (h, w) = img.shape[:2]
        aspect_ratio = float(w) / float(h)
        new_width = image_size
        new_height = int(image_size/aspect_ratio)
        if aspect_ratio <= 0 or new_height <= 0:
            print(f"Failed to load image from path: {img_path} with aspect_ration: {aspect_ratio}")
            return None
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img = img.astype('float32')
        img /= 255
        return img
    
    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")

    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []
        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1
            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    def save_dataset_info(self, df, first_index=0,
                          img_npz_path=IMG_NPZ_DIR, 
                          dataset_info_path=TRAIN_DATASET_INFO_PATH):
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        self.voc.loadVolcabulary()
        img_max_height = 0
        for index, row in df.iterrows():
            if index < first_index:
                continue
            image = row['image']
            image_name = image[:image.find(".png")]
            if os.path.isfile("{}/{}.npz".format(img_npz_path, image_name)):
                img = np.load("{}/{}.npz".format(img_npz_path, image_name))["features"]
                self.append(image_name, row['squashed_seq'], img)
                if img.shape[0] > img_max_height:
                    img_max_height = img.shape[0]
                    self.input_shape = img.shape
        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences_ids) == len(self.next_word_ids)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))
        


        # self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size
        dataset_info["output_size"] = self.voc.size
        dataset_info["size"] = self.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        dataset_info["input_shape"][0] = max(dataset_info["input_shape"][0], self.input_shape[0])
        dataset_info["input_shape"][1] = max(dataset_info["input_shape"][1], self.input_shape[1])
        dataset_info["input_shape"][2] = self.input_shape[2]

        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f)


    def save_data_in_parts(self, output_dir, num_parts):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_length = len(self.ids)
        part_size = data_length // num_parts

        for i in range(num_parts):
            start_idx = i * part_size
            end_idx = data_length if i == num_parts - 1 else (i + 1) * part_size

            np.save(os.path.join(output_dir, f'input_images_part_{i}.npy'), self.input_images[start_idx:end_idx])
            np.save(os.path.join(output_dir, f'partial_sequences_ids_part_{i}.npy'), self.partial_sequences_ids[start_idx:end_idx])
            np.save(os.path.join(output_dir, f'next_word_ids_part_{i}.npy'), self.next_word_ids[start_idx:end_idx])
            np.save(os.path.join(output_dir, f'ids_part_{i}.npy'), self.ids[start_idx:end_idx])

    def preprocess(self, df, first_index, 
                   output_dir,
                   img_npz_path=IMG_NPZ_DIR, 
                   dataset_info_path=TRAIN_DATASET_INFO_PATH, 
                   generate_binary_sequences=False,
                   num_parts=1):
        print("Preprocessing data...")
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        self.voc.loadVolcabulary()
        img_max_height = 0
        for index, row in df.iterrows():
            if index < first_index:
                continue
            image = row['image']
            image_name = image[:image.find(".png")]
            if os.path.isfile("{}/{}.npz".format(img_npz_path, image_name)):
                img = np.load("{}/{}.npz".format(img_npz_path, image_name))["features"]
                self.append(image_name, row['squashed_seq'], img)
                if img.shape[0] > img_max_height:
                    img_max_height = img.shape[0]
                    self.input_shape = img.shape

    
        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.next_word_ids = self.sparsify_labels(self.next_word_ids, self.voc)
        if generate_binary_sequences:
            self.partial_sequences_ids = self.binarize(self.partial_sequences_ids, self.voc)
        # else:
        #     self.partial_sequences_ids = self.indexify(self.partial_sequences_ids, self.voc)
        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences_ids) == len(self.next_word_ids)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))
        


        # self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size
        dataset_info["output_size"] = self.voc.size
        dataset_info["size"] = self.size
        dataset_info["num_parts"] = num_parts

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        dataset_info["input_shape"][0] = max(dataset_info["input_shape"][0], self.input_shape[0])
        dataset_info["input_shape"][1] = max(dataset_info["input_shape"][1], self.input_shape[1])
        dataset_info["input_shape"][2] = self.input_shape[2]

        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f)
        
        self.save_data_in_parts(output_dir, num_parts)


    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, equ_token_id_seq, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Dataset.show(pic)
        token_id_sequence = [self.voc.vocabulary[START_TOKEN]]
        token_id_sequence.extend(equ_token_id_seq)
        token_id_sequence.append(self.voc.vocabulary[END_TOKEN])
        suffix = [self.voc.vocabulary[PLACEHOLDER]] * CONTEXT_LENGTH
        a = np.concatenate([suffix, token_id_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context_ids = a[j:j + CONTEXT_LENGTH]
            label_id = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences_ids.append(context_ids)
            self.next_word_ids.append(label_id)

    # @staticmethod
    # def indexify(partial_sequences_ids, voc):
    #     temp = []
    #     for sequence_ids in partial_sequences_ids:
    #         sparse_vectors_sequence_id = []
    #         for token_id in sequence_ids:
    #             sparse_vectors_sequence_id.append(token_id)
    #         temp.append(np.array(sparse_vectors_sequence_id))

    #     return temp
    
    @staticmethod
    def binarize(partial_sequences_ids, voc):
        temp = []
        for sequence_ids in partial_sequences_ids:
            sparse_vectors_sequence_id = []
            for token_id in sequence_ids:
                sparse_vectors_sequence_id.append(voc.binary_vocabulary[voc.token_lookup[token_id]])
            temp.append(np.array(sparse_vectors_sequence_id))

        return temp

    @staticmethod
    def sparsify_labels(next_word_ids, voc):
        temp = []
        for label_id in next_word_ids:
            temp.append(voc.binary_vocabulary[voc.token_lookup[label_id]])

        return temp


# def is_prime(n):
#     if n <= 1:
#         return False
#     for i in range(2, int(n**0.5) + 1):
#         if n % i == 0:
#             print(i)
#             return False
#     return True


if __name__ == '__main__':
    df_train = loadData('df_train.pkl')
    print(df_train)
    dataset = Dataset()
    dataset.save_dataset_info(df_train, 0, 
                              dataset_info_path=TRAIN_DATASET_INFO_PATH)

    df_test = loadData('df_test.pkl')
    print(df_test)
    dataset = Dataset()
    dataset.save_dataset_info(df_test, 0, 
                              dataset_info_path=TEST_DATASET_INFO_PATH)
    
    df_valid = loadData('df_valid.pkl')
    print(df_valid)
    dataset = Dataset()
    dataset.save_dataset_info(df_valid, 0, 
                              dataset_info_path=VALID_DATASET_INFO_PATH)
    

    # dataset.preprocess(df_test, 0, 
    #                    output_dir=TEST_DATASET_DIR,
    #                    dataset_info_path=TEST_DATASET_INFO_PATH,
    #                    generate_binary_sequences=True,
    #                    num_parts=8)

    
