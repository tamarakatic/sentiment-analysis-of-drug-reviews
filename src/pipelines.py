from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def bag_of_words(classifier, tf_id=False):
    if tf_id:
        steps = [("vect", TfidfVectorizer())]
    else:
        steps = [("vect", CountVectorizer())]

    steps.append(("cls", classifier))
    return Pipeline(steps)
