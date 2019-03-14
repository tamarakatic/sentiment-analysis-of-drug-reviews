import preprocessor
from definitions import TEST_PROCESSED_PATH


if __name__ == '__main__':
    preprocessor = preprocessor.Preprocessor(False)
    preprocess_data = preprocessor.clean_data()
    preprocess_data[['review', 'rating']].to_csv(TEST_PROCESSED_PATH, encoding='utf-8', index=False)
