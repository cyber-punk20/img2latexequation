DATA_DIR = './dataset_tmp/training_56'
VOCAB_DIR = './dataset_tmp/step2'
IMG_DIR = './dataset_tmp/formula_images'
IMG_NPZ_DIR = './dataset_tmp/formula_images_npz'

TRAIN_DATASET_INFO_PATH = './dataset_tmp/train_dataset.json'
TEST_DATASET_INFO_PATH = './dataset_tmp/test_dataset.json'
VALID_DATASET_INFO_PATH = './dataset_tmp/valid_dataset.json'


TRAIN_DATASET_DIR = './dataset_tmp/train_dataset'
TEST_DATASET_DIR = './dataset_tmp/test_dataset'
VALID_DATASET_DIR = './dataset_tmp/valid_dataset'

MODEL_DIR = './models/'
CHECKPOINT_PATH = './models/model_checkpoint.h5'

WORD2ID_FILENAME = 'dict_vocab.pkl'
ID2WORD_FILENAME = 'dict_id2word.pkl'
BATCH_SIZE = 56
UNIQUE_IDX_OFFSET = 2000000


# CONFIG FOR MODEL
CONTEXT_LENGTH = 48
IMG_SIZE = 256
BATCH_SIZE = 56
EPOCHS = 10
STEPS_PER_EPOCH = 72000