import pandas as pd

from data.preprocessor import Preprocessor
from data.definitions import TEST_FLAIR_PATH, TRAIN_FLAIR_PATH, DEV_FLAIR_PATH


def convert_ratings_to_str(df):
    ratings = []
    for rate in df.rating:
        if rate == 0:
            ratings.append("negative")
        elif rate == 1:
            ratings.append("neutral")
        else:
            ratings.append("positive")
    df['rating'] = ratings


def preprocess_data(train=True):
    df = Preprocessor(train=train, dl=True)
    df = df.clean_data()

    convert_ratings_to_str(df)

    df.review = df.review.apply(lambda x: x[1:-1])

    df["condition"] = [cond.lower() for cond in df["condition"]]
    train_reviews = []

    for (i, row) in df.iterrows():
        train_reviews.append('{} {}'.format(row.condition, row.review))

    df["review_cond"] = train_reviews

    df.dropna(inplace=True)
    df = df[['rating', 'review_cond']].rename(columns={'rating': 'label', 'review_cond': 'text'})
    df['label'] = '__label__' + df['label'].astype(str)

    return df


if __name__ == "__main__":
    df_train = preprocess_data(True)
    df_test = preprocess_data(False)

    df_all = pd.concat([df_train, df_test])

    print("Dimension of all data: {}".format(df_all.shape))

    df_all.iloc[:int(len(df_all) * 0.8)].to_csv(
        TRAIN_FLAIR_PATH, encoding='utf-8', index=False, sep='\t', header=False)
    df_all.iloc[int(len(df_all) * 0.8):int(len(df_all) * 0.9)
                ].to_csv(DEV_FLAIR_PATH, encoding='utf-8', index=False, sep='\t', header=False)

    df_all.iloc[int(len(df_all) * 0.9):].to_csv(
        TEST_FLAIR_PATH, encoding='utf-8', index=False, sep='\t', header=False)
