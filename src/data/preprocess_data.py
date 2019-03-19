import argparse

import preprocessor
from definitions import TEST_PROCESSED_PATH, TRAIN_PROCESSED_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess = preprocessor.Preprocessor(train=args.train)
    preprocess_data = preprocess.clean_data()
    path = TRAIN_PROCESSED_PATH if args.train else TEST_PROCESSED_PATH
    preprocess_data.to_csv(path, encoding='utf-8', index=False)
