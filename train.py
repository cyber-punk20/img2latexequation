from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.keras.models import load_model


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import sys

from preprocessing import *
from pix2equation import *
from Generator import *

import os

def run(output_path=MODEL_DIR, 
        checkpoint_path=CHECKPOINT_PATH,
        dataset_info_path=TRAIN_DATASET_INFO_PATH, 
        is_memory_intensive=False, 
        pretrained_model=None,
        trained=False):
    strategy = tf.distribute.MirroredStrategy()
    np.random.seed(1234)
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)
    input_shape = tuple(dataset_info["input_shape"])
    output_size = dataset_info["output_size"]
    steps_per_epoch = int(dataset_info["size"] / BATCH_SIZE / 5)
    # steps_per_epoch = 1
    model = pix2equation(input_shape, output_size, output_path, checkpoint_path, strategy)
    


    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)
    if trained:
        model.load()
        model.compile()
        
    # model.compile()
    # df_train = loadData('df_train.pkl')
    # train_generator = Generator.data_generator(df_train, input_shape, BATCH_SIZE, verbose=False)
    # df_valid = loadData('df_valid.pkl')
    # valid_generator = Generator.data_generator(df_valid, input_shape, BATCH_SIZE)
    # model.fit(train_generator, 
    #           BATCH_SIZE,
    #           steps_per_epoch,
    #           EPOCHS,
    #           valid_generator,
    #           1)

    df_train = loadData('my_df_train.pkl')
    train_generator = Generator.data_generator_dist(df_train, input_shape, BATCH_SIZE)

    # for x, y in train_generator.take(1):
    #     print("Visual input shape:", x[0].shape)
    #     print("Textual input shape:", x[1].shape)
    #     print("Output shape:", y.shape)

    df_valid = loadData('my_df_valid.pkl')
    valid_generator = Generator.data_generator_dist(df_valid, input_shape, BATCH_SIZE)

    

    model.fit(train_generator, 
              BATCH_SIZE,
              steps_per_epoch,
              EPOCHS,
              valid_generator,
              1)

    # df_train = loadData('df_train.pkl')
    # train_dataset = Generator.generate_dataset(df_train, input_shape, BATCH_SIZE)
    # print(f'input_images: {train_dataset[0].shape}')
    # print(f'partial_sequences: {train_dataset[1].shape}')
    # print(f'next_words: {train_dataset[2].shape}')
    # df_valid = loadData('df_valid.pkl')
    # valid_dataset = Generator.generate_dataset(df_valid, input_shape, BATCH_SIZE)

    # model.fit(train_dataset[0],
    #           train_dataset[1],
    #           train_dataset[2],
    #           BATCH_SIZE,
    #           steps_per_epoch,
    #           valid_dataset[0],
    #           valid_dataset[1],
    #           valid_dataset[2],
    #           1,
    #           checkpoint)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.keras.backend.clear_session()
    tf.debugging.set_log_device_placement(False)
    tf.config.list_physical_devices('GPU')
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
        print(tf.config.list_physical_devices('GPU'))
    else:
        print("GPU is not available")
    
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    # session = InteractiveSession(config=config)
    run(trained=False)