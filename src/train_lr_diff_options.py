import pandas as pd
from time import time
from collections import defaultdict

import pipelines
import data.preprocessor as preprocessor

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def bag_of_words_pipeline():
    log_regression = pipelines.bag_of_words(
        classifier=LogisticRegression(random_state=0,
                                      solver='saga',
                                      multi_class='ovr',
                                      n_jobs=-1,
                                      max_iter=2000)
    )

    log_regression.set_params(
        vect__ngram_range=(1, 4)
    )

    return log_regression


if __name__ == "__main__":
    options = "remove_repeating_vowels"
    train_df = preprocessor.Preprocessor(train=True)
    train_df = train_df.clean_data()
    train_df.dropna(inplace=True)

    samples, labels = train_df['review'], train_df['rating']

    test_df = preprocessor.Preprocessor(train=False)
    test_df = test_df.clean_data()
    test_df.dropna(inplace=True)

    X_test, y_test = test_df['review'], test_df['rating']

    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=0.20, random_state=42
    )

    results = defaultdict(list)

    model = bag_of_words_pipeline()
    name = "BoW + LR"
    print("\n\t {}".format(name))

    results['model'].append(name)
    start = time()

    model.fit(X_train, y_train)

    end = time() - start
    print("Training finished {:.3f}s".format(end))
    results['time'].append(end)

    print("Validation model\n")
    prediction = model.predict(X_test)

    for average in ['micro', 'macro', 'weighted']:
        f1_score = metrics.f1_score(y_test, prediction, average=average)
        results['f1_{}'.format(average)].append(f1_score)

    target_names = ["Negative", "Neutral", "Positive"]

    report = metrics.classification_report(y_test, prediction, target_names=target_names)
    print("\n{}".format(report))

    with open("../reports/lg_{}_report.txt".format(options), "a+") as text_file:
        print("Report for Model: {} is \n\t {}\n".format(name, report), file=text_file)

    df = pd.DataFrame(results)
    df = df[['model', 'f1_micro', 'f1_macro', 'f1_weighted', 'time']]
    df.to_csv('../reports/lg_{}_results.csv'.format(options), index=False, float_format='%.4f')
