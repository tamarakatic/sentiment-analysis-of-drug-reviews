import pandas as pd
from time import time
from collections import defaultdict

import pipelines
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from data.definitions import TRAIN_PATH, TEST_PATH

joblib_model = "../models/finilized_model.pkl"


def bag_of_words_pipeline():
    log_regression = pipelines.bag_of_words(
        classifier=LogisticRegression(random_state=0,
                                      solver='saga',
                                      multi_class='ovr',
                                      n_jobs=-1,
                                      max_iter=2000)
    )

    log_regression_tfidf = pipelines.bag_of_words(
        classifier=LogisticRegression(random_state=0,
                                      solver='saga',
                                      multi_class='ovr',
                                      n_jobs=-1,
                                      max_iter=2000),
        tf_idf=True
    )

    multinomial_nb = pipelines.bag_of_words(
        classifier=MultinomialNB()
    )

    multinomial_nb_tfidf = pipelines.bag_of_words(
        classifier=MultinomialNB(),
        tf_idf=True
    )

    linear_svc = pipelines.bag_of_words(
        classifier=LinearSVC(multi_class='ovr', max_iter=2000, random_state=0)
    )

    linear_svc_tfidf = pipelines.bag_of_words(
        classifier=LinearSVC(multi_class='ovr', max_iter=2000, random_state=0),
        tf_idf=True
    )

    # tune: max_depth, min_child_weight, n_estimatorss
    xgb = pipelines.bag_of_words(
        classifier=XGBClassifier(
            learning_rate=0.2, n_estimators=1000, max_depth=5, objective='multi:softprob',
            n_class=3, n_jobs=-1, random_state=0)
    )

    xgb_tfidf = pipelines.bag_of_words(
        classifier=XGBClassifier(
            learning_rate=0.2, n_estimators=1000, max_depth=5, objective='multi:softprob',
            n_class=3, n_jobs=-1, random_state=0),
        tf_idf=True
    )

    sgd_classifier = pipelines.bag_of_words(
        classifier=SGDClassifier(max_iter=2000, n_jobs=-1, random_state=0)
    )

    sgd_classifier_tfidf = pipelines.bag_of_words(
        classifier=SGDClassifier(max_iter=2000, n_jobs=-1, random_state=0),
        tf_idf=True
    )

    bow_pipelines = [
        ("BoW + LR", log_regression),
        ("BoW + LR + TFIDF", log_regression_tfidf),
        ("BoW + MNB", multinomial_nb),
        ("BoW + MNB + TFIDF", multinomial_nb_tfidf),
        ("BoW + LinearSVC", linear_svc),
        ("BoW + LinearSVC + TFIDF", linear_svc_tfidf),
        ("BoW + XGBoost", xgb),
        ("BoW + XGBoost + TFIDF", xgb_tfidf),
        ("BoW + SGDClassifier", sgd_classifier),
        ("BoW + SGDClassifier + TFIDF", sgd_classifier_tfidf)
    ]

    for name, model in bow_pipelines:
        model.set_params(
            vect__ngram_range=(1, 4)
        )

        yield (name, model)


if __name__ == "__main__":
    preprocess_data = pd.read_csv(TRAIN_PATH)
    preprocess_data.dropna(inplace=True)
    samples, labels = preprocess_data['review'], preprocess_data['rating']

    test_df = pd.read_csv(TEST_PATH)
    test_df.dropna(inplace=True)

    X_test, y_test = test_df['review'], test_df['rating']
    target_names = ["Negative", "Neutral", "Positive"]
    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=0.20, random_state=42
    )

    results = defaultdict(list)

    for name, model in bag_of_words_pipeline():
        print("\n\t".format(name))

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

        report = metrics.classification_report(y_test, prediction, target_names=target_names)
        print("\n{}".format(report))

        with open("../reports/report.txt", "a+") as text_file:
            print("Report for Model: {} is \n\t {}\n".format(name, report), file=text_file)

        with open("../models/{}_model.pkl".format(name), 'wb') as fp:
            joblib.dump(model, fp)

    df = pd.DataFrame(results)
    df = df[['model', 'f1_micro', 'f1_macro', 'f1_weighted', 'time']]
    df.to_csv('../reports/results.csv', index=False, float_format='%.4f')
