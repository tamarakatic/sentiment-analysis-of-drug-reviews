from elmo.dataset_reader import ElmoDatasetReader
from data.definitions import DEV_BERT_PATH, TRAIN_BERT_PATH, TEST_BERT_PATH

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.iterators import BucketIterator

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32


def spacy_tokenizer(sentence: str):
    spacy_splitter = SpacyWordSplitter(language='en_core_web_sm', pos_tags=False)
    list_of_tokens = spacy_splitter.split_words(sentence)[:MAX_SEQ_LENGTH]
    return [w.text for w in list_of_tokens]


def dataset_reader(train=True):
    token_indexer = ELMoTokenCharactersIndexer()
    reader = ElmoDatasetReader(
        tokenizer=spacy_tokenizer,
        token_indexers={"tokens": token_indexer}
    )

    if train:
        train_dataset = reader.read(TRAIN_BERT_PATH)
        dev_dataset = reader.read(DEV_BERT_PATH)
        return train_dataset, dev_dataset
    else:
        test_dataset = reader.read(TEST_BERT_PATH)
        return test_dataset


def data_iterator(vocabulary):
    iterator = BucketIterator(batch_size=BATCH_SIZE,
                              sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocabulary)
    return iterator
