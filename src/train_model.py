import pandas as pd
from time import time

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from data.definitions import TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',  OneVsRestClassifier(LogisticRegression(random_state=0, solver='sag', max_iter=2000))),
])

parameters = {
    'tfidf__use_idf': (True, False),
}

if __name__ == "__main__":
    preprocess_data = pd.read_csv(TRAIN_PROCESSED_PATH)
    preprocess_data.dropna(inplace=True)
    samples, labels = preprocess_data['review'], preprocess_data['rating']

    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=0.20, random_state=42
    )

    grid_search = GridSearchCV(pipeline,
                               parameters,
                               scoring="f1_macro",
                               cv=5,
                               verbose=1,
                               n_jobs=-1)

    print("Performing Grid Search. . .")
    print("Pipeline: {}".format(name for name, _ in pipeline.steps))
    print("Parameters: {}".format(parameters))

    start = time()
    grid_search.fit(X_val, y_val)

    print("Training finished {:.3f}s".format(time() - start))
    print("Best score is: {}".format(grid_search.best_score_))
    print("Best parameters are:")

    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t{}: {}".format(param_name, best_params[param_name]))

    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    print("Validation model\n")
    test_df = pd.read_csv(TEST_PROCESSED_PATH)
    test_df.dropna(inplace=True)
    X_test, y_test = test_df['review'], test_df['rating']

    prediction = grid_search.predict(X_test)

    target_names = ["Negative", "Neutral", "Positive"]
    report = metrics.classification_report(y_test, prediction, target_names=target_names)
    print("\n{}".format(report))
