import os

file_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(file_path))

DATA_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/')
DATA_BERT = os.path.join(ROOT_PATH, 'data/bert/tmp')

BERT_PRETRAINED_PATH = os.path.join(ROOT_PATH, 'models/pretrain/uncased_L-12_H-768_A-12/')
PRETRAINED_BERT_CACHE = os.path.join(ROOT_PATH, 'models/pretrain/uncased_L-12_H-768_A-12/cache/')

PRETRAINED_ELMO = os.path.join(ROOT_PATH, 'models/pretrain/elmo/')

DEV_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/dev.csv')
TRAIN_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/train.csv')
TEST_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/test.csv')

TRAIN_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_train.tsv')
TEST_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_test.tsv')

TRAIN_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/train.csv')
TEST_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/test.csv')
DEV_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/dev.csv')

OPTION_FILE = os.path.join(ROOT_PATH, 'models/pretrain/elmo/elmo_small_options.json')
WEIGHT_FILE = os.path.join(
    ROOT_PATH, 'models/pretrain/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
