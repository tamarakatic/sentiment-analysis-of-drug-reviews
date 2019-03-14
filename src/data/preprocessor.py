import pandas as pd
import re

from definitions import TRAIN_RAW_PATH, TEST_RAW_PATH
from patterns import NEGATIVE_CONSTRUCTS, POSITIVE_EMOTICONS, NEGATIVE_EMOTICONS

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from string import punctuation, whitespace
from spellchecker import SpellChecker
import language_check
from pipe import Pipe


pd.options.mode.chained_assignment = None


class Preprocessor:

    def __init__(self, train=True):
        self._data_frame = load_data(train)

    def clean_data(self):
        new_data_frame = self._data_frame.copy()
        new_data_frame['review'] = preprocess_data(new_data_frame.iloc[:1, :])
        new_data_frame['rating'] = convert_ratings(new_data_frame['rating'])
        return new_data_frame.iloc[:1, :]


def load_data(train):
    data_frame = pd.read_csv(TRAIN_RAW_PATH if train else TEST_RAW_PATH, sep='\t', encoding="utf-8")
    data_frame.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    data_frame.fillna("", inplace=True)
    data_frame.drop(['date', 'usefulCount'], axis=1, inplace=True)
    data_frame.update(data_frame.groupby('drugName')['rating'].mean().reset_index())
    return data_frame


def convert_ratings(data):
    ratings = []
    for rate in data:
        if rate <= 4:
            ratings.append(1)
        elif rate > 4 and rate < 7:
            ratings.append(0)
        else:
            ratings.append(2)

    return ratings


def preprocess_data(df):
    df['review'] = [clean_review(sentence) for sentence in df['review']]
    return (df['review'])


def clean_review(sentence):
    return sentence \
        | split_attached_words \
        | correct_spelling \
        | correct_grammar \
        | remove_repeating_vowels \
        | convert_text_to_lowercase \
        | remove_digits \
        | remove_punctuation \
        | remove_stopwords_and_whitespaces \
        | remove_emails \
        | remove_urls \
        | handle_negations \
        | replace_emoticons_with_tags \
        | stem \
        | lemmatize


@Pipe
def split_attached_words(sentence):
    return " ".join(re.findall("[A-Z][^A-Z]*", sentence))


@Pipe
def correct_spelling(sentence):
    spell = SpellChecker()
    return " ".join([spell.correction(word) for word in sentence.split()])


@Pipe
def correct_grammar(sentence):
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(sentence)
    return language_check.correct(sentence, matches)


@Pipe
def remove_repeating_vowels(sentence):
    return re.sub(r"(.)\1+", r"\1\1", sentence)


@Pipe
def convert_text_to_lowercase(sentence):
    return sentence.lower()


@Pipe
def remove_digits(sentence):
    return re.sub(r"\d+", "", sentence)


@Pipe
def remove_punctuation(sentence):
    return sentence.translate(str.maketrans("", "", punctuation))


@Pipe
def remove_stopwords_and_whitespaces(sentence):
    stop_words = set(stopwords.words("english") + list(whitespace))
    return " ".join([word for word in sentence.split() if word not in stop_words])


@Pipe
def remove_emails(sentence):
    return re.sub(r"\S*@\S*\s?", "", sentence)


@Pipe
def remove_urls(sentence):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)


@Pipe
def handle_negations(sentence):
    return " ".join(["not" if word in NEGATIVE_CONSTRUCTS else word for word in sentence.split()])


def check_emoticons(word):
    if word in POSITIVE_EMOTICONS:
        return "positive"
    elif word in NEGATIVE_EMOTICONS:
        return "negative"
    else:
        return word


@Pipe
def replace_emoticons_with_tags(sentence):
    return " ".join([check_emoticons(word) for word in sentence.split()])


def get_word_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


@Pipe
def stem(sentence):
    stemmer = SnowballStemmer("english")
    return " ".join(stemmer.stem(word) for word in sentence.split())


@Pipe
def lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word, get_word_pos(word)) for word in sentence.split()])
