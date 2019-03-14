import os

file_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(file_path))

TRAIN_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_train.tsv')
TEST_RAW_PATH = os.path.join(ROOT_PATH, 'data/raw/drugs_test.tsv')

TRAIN_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/drugs_train.csv')
TEST_PROCESSED_PATH = os.path.join(ROOT_PATH, 'data/processed/drugs_test.csv')
