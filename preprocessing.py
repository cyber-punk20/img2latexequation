import pandas as pd
import numpy as np
import os
import sys

from constants import *
from utils import *

PLACEHOLDER = " "
START_TOKEN = '\\bos'
END_TOKEN = '\\eos'

def loadData(filename: str)-> pd.DataFrame:
    df = pd.read_pickle(os.path.join(DATA_DIR, filename))
    # print(df['seq_len'].equals(df['squashed_len']))
    return df

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

        self.input_images = []

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


    def load(self, img_path):
        print("Loading data...")
        self.voc.loadVolcabulary()
        for f in os.listdir(img_path):
            if f.find('.png') != -1:
                img = Dataset.get_preprocessed_img("{}/{}".format(img_path, f), image_size=IMG_SIZE)
        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
    
    def append(self, sample_id, equ_token_seq, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Dataset.show(pic)
        token_sequence = [START_TOKEN]
        token_sequence.extend(equ_token_seq)
        token_sequence.append(END_TOKEN)
        suffix = [PLACEHOLDER] * CONTEXT_LENGTH
        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)
        
    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

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
    # df_train = loadData('df_train.pkl')
    # print(df_train.sort_index())
    # df_valid = loadData('df_valid.pkl')
    # print(df_valid.sort_index())
    convert_imgs_npz(PNG_DIR, PNG_NPZ_DIR)
    # df_test = loadData('df_test.pkl')
    # print(df_test.sort_index())
    # vocabulary = Vocabulary()
    # vocabulary.loadVolcabulary()
    # print(vocabulary.vocabulary)
    # # print(vocabulary.token_lookup)
    # # print(is_prime(1719233839))
    # # check_png_size(PNG_DIR)
    # dataset = Dataset()

    
