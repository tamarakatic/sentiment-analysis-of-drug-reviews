import os

file_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(file_path))

DATA_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/')
DATA_BERT = os.path.join(ROOT_PATH, 'data/bert/tmp')
OUTPUT_BERT_DIR = os.path.join(ROOT_PATH, 'data/bert/tmp/class/output')

BERT_PRETRAINED_PATH = os.path.join(ROOT_PATH, 'models/pretrain/uncased_L-12_H-768_A-12/')
PRETRAINED_BERT_CACHE = os.path.join(ROOT_PATH, 'models/pretrain/uncased_L-12_H-768_A-12/cache/')

DEV_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/dev.csv')
TRAIN_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/train.csv')
TEST_BERT_PATH = os.path.join(ROOT_PATH, 'data/bert/test.csv')

TRAIN_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_train.tsv')
TEST_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_test.tsv')

TRAIN_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/drugs_train.csv')
TEST_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/drugs_test.csv')
