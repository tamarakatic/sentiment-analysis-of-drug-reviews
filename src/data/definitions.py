import os

file_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(file_path))

DATA_PATH = os.path.join(ROOT_PATH, 'data/processed/')
DATA_BERT = os.path.join(ROOT_PATH, 'data/processed/tmp')

BERT_PRETRAINED_PATH = os.path.join(ROOT_PATH, 'models/pretrain/uncased_L-12_H-768_A-12/')
OUTPUT_BERT_DIR = os.path.join(ROOT_PATH, 'models/pretrain/uncased_L-12_H-768_A-12/cache/')

PRETRAINED_ELMO = os.path.join(ROOT_PATH, 'models/pretrain/elmo/')
PRETRAINED_BERT = os.path.join(ROOT_PATH, 'models/pretrain/bert/')

PRETRAINED_FLAIR = os.path.join(ROOT_PATH, 'models/pretrain/flair/')
FLAIR_LOSS = os.path.join(ROOT_PATH, 'models/pretrain/flair/loss.tsv')
FLAIR_WEIGHTS = os.path.join(ROOT_PATH, 'models/pretrain/flair/weights.txt')

DEV_PATH = os.path.join(ROOT_PATH, 'data/processed/dev.csv')
TRAIN_PATH = os.path.join(ROOT_PATH, 'data/processed/train.csv')
TEST_PATH = os.path.join(ROOT_PATH, 'data/processed/test.csv')

TEST_FLAIR_PATH = os.path.join(ROOT_PATH, 'data/processed/flair_test.csv')
TRAIN_FLAIR_PATH = os.path.join(ROOT_PATH, 'data/processed/flair_train.csv')
DEV_FLAIR_PATH = os.path.join(ROOT_PATH, 'data/processed/flair_dev.csv')

TRAIN_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_train.tsv')
TEST_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_test.tsv')

OPTION_FILE = os.path.join(ROOT_PATH, 'models/pretrain/elmo/elmo_small_options.json')
WEIGHT_FILE = os.path.join(
    ROOT_PATH, 'models/pretrain/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
