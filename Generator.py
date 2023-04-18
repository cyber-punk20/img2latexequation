from __future__ import print_function

import numpy as np
import tensorflow as tf

from preprocessing import *
from constants import *
from tensorflow.keras.utils import Sequence


class Generator:
    @staticmethod
    def data_generator(df, input_shape, batch_size,  
                       img_npz_path=IMG_NPZ_DIR,
                       verbose=False, loop_only_one=False):
        dataset = Dataset()
        dataset.voc.loadVolcabulary()
        dataset.voc.create_binary_representation()

        while 1:
            batch_input_images = []
            batch_partial_sequences = []
            batch_next_words = []
            sample_in_batch_counter = 0
            i = 0
            for index, row in df.iterrows():
                image = row['image']
                image_name = image[:image.find(".png")]
                if os.path.isfile("{}/{}.npz".format(img_npz_path, image_name)):
                    img = np.load("{}/{}.npz".format(img_npz_path, image_name))["features"]
                    pad_width = ((0, input_shape[0] - img.shape[0]), (0, 0), (0, 0))
                    img = np.pad(img, pad_width, mode='constant')
                else:
                    continue
                equ_token_id_seq = row['squashed_seq']
                token_id_sequence = [dataset.voc.vocabulary[START_TOKEN]]
                token_id_sequence.extend(equ_token_id_seq)
                token_id_sequence.append(dataset.voc.vocabulary[END_TOKEN])
                suffix = [dataset.voc.vocabulary[PLACEHOLDER]] * CONTEXT_LENGTH

                a = np.concatenate([suffix, token_id_sequence])
                for j in range(0, len(a) - CONTEXT_LENGTH):
                    context_ids = a[j:j + CONTEXT_LENGTH]
                    label_id = a[j + CONTEXT_LENGTH]

                    batch_input_images.append(img)
                    batch_partial_sequences.append(context_ids)
                    batch_next_words.append(label_id)
                    sample_in_batch_counter += 1

                    if sample_in_batch_counter == batch_size or (loop_only_one and i == len(df) - 1):
                        if verbose:
                            print("Generating sparse vectors...")
                        batch_next_words = Dataset.sparsify_labels(batch_next_words, dataset.voc)
                        batch_partial_sequences = Dataset.binarize(batch_partial_sequences, dataset.voc)

                        if verbose:
                            print("Convert arrays...")
                        batch_input_images = np.array(batch_input_images)
                        batch_partial_sequences = np.array(batch_partial_sequences)
                        batch_next_words = np.array(batch_next_words)

                        if verbose:
                            print("Yield batch")
                            print(batch_input_images.shape)
                            print(batch_partial_sequences.shape)
                            print(batch_next_words.shape)
                        yield ([batch_input_images, batch_partial_sequences], batch_next_words)

                        batch_input_images = []
                        batch_partial_sequences = []
                        batch_next_words = []
                        sample_in_batch_counter = 0
                i += 1
    @staticmethod
    def create_distributed_dataset_from_generator(df, input_shape, batch_size, img_npz_path=IMG_NPZ_DIR, verbose=False):
        dataset = Dataset()
        dataset.voc.loadVolcabulary()
        dataset.voc.create_binary_representation()
        voc_size = dataset.voc.size
        output_signature = (
            tf.TensorSpec(shape=(None, input_shape[0], input_shape[1], input_shape[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(None, CONTEXT_LENGTH, voc_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None, voc_size), dtype=tf.float32)
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: Generator.data_generator(df, input_shape, batch_size, img_npz_path, verbose),
            output_signature = output_signature
        )

        # Distribute the dataset using the strategy
        # dist_dataset = strategy.experimental_distribute_dataset(dataset)
        # return dist_dataset
        return dataset
