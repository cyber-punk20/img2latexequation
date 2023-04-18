from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import sys

from preprocessing import *
from pix2equation import *
from Generator import *


from tensorflow.keras import backend as K

import os

def run(output_path=MODEL_DIR, 
        checkpoint_path=CHECKPOINT_PATH,
        dataset_info_path=TRAIN_DATASET_INFO_PATH, 
        is_memory_intensive=False, 
        pretrained_model=None):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    np.random.seed(1234)
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)
    input_shape = tuple(dataset_info["input_shape"])
    output_size = dataset_info["output_size"]
    steps_per_epoch = int(dataset_info["size"] / BATCH_SIZE)

    model = pix2equation(input_shape, output_size, output_path, strategy)
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min')


    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)


    df_train = loadData('df_train.pkl')
    train_generator = Generator.data_generator(df_train, input_shape, BATCH_SIZE, verbose=False)
    # dist_train_dataset = strategy.experimental_distribute_dataset(Generator.create_distributed_dataset_from_generator(df_train, input_shape, BATCH_SIZE, verbose=False))


    df_valid = loadData('df_valid.pkl')
    valid_generator = Generator.data_generator(df_valid, input_shape, BATCH_SIZE)
    


    model.fit(train_generator, 
              BATCH_SIZE,
              steps_per_epoch,
              valid_generator,
              2,
              checkpoint)


    # df_train = loadData('df_train.pkl')
    # dist_train_dataset = strategy.experimental_distribute_dataset(Generator.create_distributed_dataset_from_generator(df_train, input_shape, BATCH_SIZE, verbose=False))
    # df_valid = loadData('df_valid.pkl')
    # dist_valid_dataset = strategy.experimental_distribute_dataset(Generator.create_distributed_dataset_from_generator(df_valid, input_shape, BATCH_SIZE, verbose=False))

    # model.fit(dist_train_dataset, 
    #           BATCH_SIZE,
    #           steps_per_epoch,
    #           dist_valid_dataset,
    #           2,
    #           checkpoint)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    K.clear_session()
    tf.debugging.set_log_device_placement(False)
    tf.config.list_physical_devices('GPU')
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
        print(tf.config.list_physical_devices('GPU'))
    else:
        print("GPU is not available")
    run()