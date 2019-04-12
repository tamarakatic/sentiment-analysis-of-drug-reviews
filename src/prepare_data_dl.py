from data.definitions import TRAIN_PATH, TEST_PATH, DEV_PATH
from data.preprocessor import Preprocessor


def prepare_data(train=True):
    df = Preprocessor(train=train, dl=True)
    df = df.clean_data()

    df.review = df.review.apply(lambda x: x[1:-1])
    df["condition"] = [cond.lower() for cond in df["condition"]]
    train_reviews = []

    for (i, row) in df.iterrows():
        train_reviews.append('{} {}'.format(row.condition, row.review))

    df["review_cond"] = train_reviews
    df = df[df.review_cond.str.split().map(len) <= 128]
    return df


if __name__ == "__main__":
    df_train = prepare_data(True)

    train_len = int(len(df_train) * 0.8)

    df_train[:train_len].to_csv(TRAIN_PATH, encoding='utf-8', index=False)
    df_train[train_len:].to_csv(DEV_PATH, encoding='utf-8', index=False)

    df_test = prepare_data(False)
    df_test.to_csv(TEST_PATH, encoding='utf-8', index=False)
