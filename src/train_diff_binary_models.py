import pandas as pd
from time import time
from collections import defaultdict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import warnings
from termcolor import colored

import pipelines
from data.definitions import TRAIN_PATH, TEST_PATH

warnings.filterwarnings('ignore')


def get_negative_positive_sentiment(df):
    negative_sentiment = df[(df.rating == 0)]
    positive_sentiment = df[(df.rating == 1)].sample(frac=1)[:len(negative_sentiment)]
    new_df = pd.concat([negative_sentiment, positive_sentiment])
    return new_df['review'], new_df['rating']


def train_data():
    dummy_clf = pipelines.bag_of_words(
        classifier=DummyClassifier(random_state=0,
                                   strategy="stratified")
    )

    log_regression = pipelines.bag_of_words(
        classifier=LogisticRegression(random_state=0,
                                      n_jobs=-1,
                                      max_iter=2000)
    )

    log_regression_tfidf = pipelines.bag_of_words(
        classifier=LogisticRegression(random_state=0,
                                      n_jobs=-1,
                                      max_iter=2000),
        tf_idf=True
    )

    linear_svc = pipelines.bag_of_words(
        classifier=LinearSVC(max_iter=2000, random_state=0)
    )

    linear_svc_tfidf = pipelines.bag_of_words(
        classifier=LinearSVC(max_iter=2000, random_state=0),
        tf_idf=True
    )

    # svc = pipelines.bag_of_words(
    #     classifier=SVC(gamma='scale', kernel='rbf')
    # )

    bow_pipelines = [
        ("Binary: BoW + Dummy", dummy_clf),
        ("Binary: BoW + LR", log_regression),
        ("Binary: BoW + LR + TFIDF", log_regression_tfidf),
        ("Binary: BoW + LinearSVC", linear_svc),
        ("Binary: BoW + LinearSVC + TFIDF", linear_svc_tfidf)
        # ("Binary: BoW + SVC", svc)
    ]
    for name, model in bow_pipelines:
        model.set_params(
            vect__ngram_range=(1, 4)
        )

        yield (name, model)


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH).sample(frac=1)
    train_df.dropna(inplace=True)
    samples, labels = get_negative_positive_sentiment(train_df)

    test_df = pd.read_csv(TEST_PATH).sample(frac=1)
    test_df.dropna(inplace=True)
    X_test, y_test = get_negative_positive_sentiment(test_df)

    target_names = ["Negative", "Positive"]
    results = defaultdict(list)

    for name, model in train_data():
        print("\n\t".format(name))

        results['model'].append(name)
        start = time()

        model.fit(samples, labels)

        end = time() - start
        print("Training finished {:.3f}s\n".format(end))
        results['time'].append(end)

        prediction = model.predict(X_test)

        print("\n{}\n{}".format(colored('\t\tConfusion matrix', 'blue'),
                                confusion_matrix(y_test, prediction)))

        f1_score = metrics.f1_score(y_test, prediction)
        accuracy = metrics.accuracy_score(y_test, prediction)
        results['accuracy'].append(accuracy)

        print("\n{}\n".format(colored('\t\t{}'.format(name), 'red')))
        print("{} {}".format(colored('Accuracy: ', 'green'), accuracy))
        report = metrics.classification_report(y_test, prediction, target_names=target_names)
        print("\n{}\n{}".format(colored('\t\tClassification report', 'yellow'), report))

        with open("../reports/report.txt", "a+") as text_file:
            print("Report for Model: {} is \n\t {}\n".format(name, report), file=text_file)

    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'time']]
    results_df.to_csv('../reports/binary_results.csv', index=False, float_format='%.4f')
