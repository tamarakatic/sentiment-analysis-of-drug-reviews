from data.definitions import TRAIN_BERT_PATH, TEST_BERT_PATH, DEV_BERT_PATH
from data.preprocessor import Preprocessor

if __name__ == "__main__":
    df_train = Preprocessor(True)
    df_train = df_train.clean_data()

    df_train = df_train[df_train.review.str.split().map(len) <= 128]
    df_train["review"] = [review.lower() for review in df_train["review"]]

    train_len = int(len(df_train) * 0.2)

    df_train[:train_len].to_csv(DEV_BERT_PATH, encoding='utf-8', index=False)
    df_train[train_len:].to_csv(TRAIN_BERT_PATH, encoding='utf-8', index=False)

    df_test = Preprocessor(False)
    df_test = df_test.clean_data()
    df_test = df_test[df_test.review.str.split().map(len) <= 128]
    df_test["review"] = [review.lower() for review in df_test["review"]]

    df_test.to_csv(TEST_BERT_PATH, encoding='utf-8', index=False)
