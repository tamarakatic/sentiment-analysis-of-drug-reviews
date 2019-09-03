import pandas as pd
import numpy as np
from time import time
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import warnings
from termcolor import colored

import pipelines
from data.definitions import TRAIN_RAW_PATH, TEST_RAW_PATH

warnings.filterwarnings('ignore')

sent_analyzer = SentimentIntensityAnalyzer()


def evaluate_polarity(compound_score):
    if compound_score >= 0.05:
        return 1
    elif (compound_score > -0.05) and (compound_score < 0.05):
        return 2
    else:
        return 0


def convert_sentiment(df):
    sentiments = []

    for review in df.review:
        sentiments.append(evaluate_polarity(sent_analyzer.polarity_scores(review).get('compound')))

    df["sentiment"] = pd.Series(data=np.asarray(sentiments))
    pos_neg_sentiment = df[(df["sentiment"] == 0) | (df["sentiment"] == 1)]
    return pos_neg_sentiment['review'], pos_neg_sentiment['sentiment']


def train_data():
    log_regression = pipelines.bag_of_words(
        classifier=LogisticRegression(random_state=0,
                                      n_jobs=-1,
                                      max_iter=2000,
                                      verbose=1)
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

    svc = pipelines.bag_of_words(
        classifier=SVC(gamma='scale', kernel='rbf')
    )

    dummy_clf = pipelines.bag_of_words(
        classifier=DummyClassifier(random_state=0,
                                   strategy="stratified")
    )

    bow_pipelines = [
        # ("Vader: BoW + LR", log_regression),
        # ("Vader: BoW + Dummy", dummy_clf),
        # ("Vader: BoW + LR + TFIDF", log_regression_tfidf),
        # ("Vader: BoW + LinearSVC", linear_svc),
        # ("Vader: BoW + LinearSVC + TFIDF", linear_svc_tfidf),
        ("Vader: BoW + SVC", svc)
    ]
    for name, model in bow_pipelines:
        model.set_params(
            vect__ngram_range=(1, 4)
        )

        yield (name, model)


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_RAW_PATH, sep='\t', encoding="utf-8")[:50000]
    train_df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    train_df.dropna(inplace=True)

    samples, labels = convert_sentiment(train_df)

    test_df = pd.read_csv(TEST_RAW_PATH, sep='\t', encoding="utf-8")[:15000]
    test_df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    test_df.dropna(inplace=True)

    X_test, y_test = convert_sentiment(test_df)
    target_names = ["Negative", "Positive"]
    # X_train, X_val, y_train, y_val = train_test_split(
    #     samples, labels, test_size=0.20, random_state=42
    # )

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

        confusion_matrix = confusion_matrix(y_test, prediction)
        print("\n{}\n{}".format(colored('\t\tConfusion matrix', 'blue'), confusion_matrix))

        f1_score = metrics.f1_score(y_test, prediction)
        accuracy = metrics.accuracy_score(y_test, prediction)
        results['accuracy'].append(accuracy)

        print("\n{}\n".format(colored('\t\t{}'.format(name), 'red')))
        print("{} {}".format(colored('Accuracy: ', 'green'), accuracy))
        report = metrics.classification_report(y_test, prediction, target_names=target_names)
        print("\n{}\n{}".format(colored('\t\tClassification report', 'yellow'), report))

        with open("../reports/report.txt", "a+") as text_file:
            print("Report for Model: {} is \n\t {}\n".format(name, report), file=text_file)

        # with open("../models/{}_model.pkl".format(name), 'wb') as fp:
        #     joblib.dump(model, fp)

    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'time']]
    results_df.to_csv('../reports/vader_results.csv', index=False, float_format='%.4f')
