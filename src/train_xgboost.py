import pandas as pd
from time import time

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from xgboost.sklearn import XGBClassifier
from data.definitions import TRAIN_PATH, TEST_PATH


pipeline = Pipeline([
    ('vect', CountVectorizer(1, 4)),
    ('clf', XGBClassifier(n_estimators=1000, objective='multi:softprob',
                          n_class=3, n_jobs=-1, random_state=0)),
])

params = {
    'learning_rate': [0.2, 0.1, 0.01],
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}

if __name__ == "__main__":
    preprocess_data = pd.read_csv(TRAIN_PATH)
    preprocess_data.dropna(inplace=True)
    samples, labels = preprocess_data['review'], preprocess_data['rating']

    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=0.20, random_state=42
    )

    grid_search = GridSearchCV(pipeline,
                               params,
                               scoring="f1_macro",
                               cv=5,
                               verbose=1,
                               n_jobs=-1)

    print("Performing Grid Search. . .")
    print("Pipeline: {}".format(name for name, _ in pipeline.steps))
    print("Parameters: {}".format(params))

    start = time()
    grid_search.fit(X_val, y_val)

    print("Training finished {:.3f}s".format(time() - start))
    print("Best score is: {}".format(grid_search.best_score_))
    print("Best parameters are:")

    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t{}: {}".format(param_name, best_params[param_name]))

    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    print("Validation model\n")
    test_df = pd.read_csv(TEST_PATH)
    test_df.dropna(inplace=True)
    X_test, y_test = test_df['review'], test_df['rating']

    prediction = grid_search.predict(X_test)

    target_names = ["Negative", "Neutral", "Positive"]
    report = metrics.classification_report(y_test, prediction, target_names=target_names)
    print("\n{}".format(report))

    with open("../reports/xgb_report.txt", "a+") as text_file:
        print("Report for XGBoost model: \n\t {}\n".format(report), file=text_file)
