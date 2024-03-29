DATA_DIR = './dataset_tmp/training_56'
VOCAB_DIR = './dataset_tmp/step2'
IMG_DIR = './dataset_tmp/formula_images'
IMG_NPZ_DIR = './dataset_tmp/formula_images_npz_128'

TRAIN_DATASET_INFO_PATH = './dataset_tmp/train_dataset_128.json'
TEST_DATASET_INFO_PATH = './dataset_tmp/test_dataset_128.json'
VALID_DATASET_INFO_PATH = './dataset_tmp/valid_dataset_128.json'


TRAIN_DATASET_DIR = './dataset_tmp/train_dataset'
TEST_DATASET_DIR = './dataset_tmp/test_dataset'
VALID_DATASET_DIR = './dataset_tmp/valid_dataset'

MODEL_DIR = './models/'
CHECKPOINT_PATH = './models/model_checkpoint'

WORD2ID_FILENAME = 'dict_vocab.pkl'
ID2WORD_FILENAME = 'dict_id2word.pkl'
BATCH_SIZE = 512
UNIQUE_IDX_OFFSET = 2000000


PLACEHOLDER = " "
START_TOKEN = '\\bos'
END_TOKEN = '\\eos'

EVAL_BLUE_SCORE_PATH = 'blue_scores.txt'


# CONFIG FOR MODEL
CONTEXT_LENGTH = 48
IMG_SIZE = 128
EPOCHS = 10
STEPS_PER_EPOCH = 72000


MAX_OUTPUT_LENGTH=150