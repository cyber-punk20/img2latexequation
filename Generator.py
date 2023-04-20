from __future__ import print_function

import numpy as np
import tensorflow as tf

from preprocessing import *
from constants import *
from tensorflow.keras.utils import Sequence


class Generator:
    @staticmethod
    def single_example_generator(df, input_shape, img_npz_path=IMG_NPZ_DIR):
        dataset = Dataset()
        dataset.voc.loadVolcabulary()
        dataset.voc.create_binary_representation()

        while True:
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
                    context_ids = Dataset.binarize_context_ids(context_ids, dataset.voc)
                    label_id = a[j + CONTEXT_LENGTH]
                    label_id = Dataset.sparsify_label(label_id, dataset.voc)

                    yield ((np.array(img), np.array(context_ids)), np.array(label_id))

    @staticmethod
    def data_generator_dist(df, input_shape, batch_size, img_npz_path=IMG_NPZ_DIR):
        voc = Dataset().voc
        voc.loadVolcabulary()
        voc.create_binary_representation()
        dataset = tf.data.Dataset.from_generator(
            lambda: Generator.single_example_generator(df, input_shape, img_npz_path),
            output_signature=(
                (tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                 tf.TensorSpec(shape=(CONTEXT_LENGTH, len(voc.vocabulary)), dtype=tf.float32)),
                 tf.TensorSpec(shape=(len(voc.vocabulary),), dtype=tf.float32)
            )
        )

        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
    @staticmethod
    def generate_dataset(df, input_shape, batch_size,  
                         img_npz_path=IMG_NPZ_DIR):
        dataset = Dataset()
        dataset.voc.loadVolcabulary()
        dataset.voc.create_binary_representation()

        input_images = []
        partial_sequences = []
        next_words = []

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

                input_images.append(img)
                partial_sequences.append(context_ids)
                next_words.append(label_id)

        next_words = Dataset.sparsify_labels(next_words, dataset.voc)
        partial_sequences = Dataset.binarize(partial_sequences, dataset.voc)
        input_images = np.array(input_images)
        partial_sequences = np.array(partial_sequences)
        next_words = np.array(next_words)
        return (input_images, partial_sequences, next_words)
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
