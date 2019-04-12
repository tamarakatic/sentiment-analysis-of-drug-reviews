from tl_allennlp.dataset_reader import TransferLearnDatasetReader
from data.definitions import TEST_PATH, TRAIN_PATH, DEV_PATH

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.iterators import BucketIterator

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32


def spacy_tokenizer(sentence: str):
    spacy_splitter = SpacyWordSplitter(language='en_core_web_sm', pos_tags=False)
    list_of_tokens = spacy_splitter.split_words(sentence)[:MAX_SEQ_LENGTH]
    return [w.text for w in list_of_tokens]


def bert_indexer():
    bert_token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-uncased",
        max_pieces=MAX_SEQ_LENGTH,
        do_lowercase=True
    )
    return bert_token_indexer


def bert_tokenizer(sentence: str):
    token_indexer = bert_indexer()
    return token_indexer.wordpiece_tokenizer(sentence)[:MAX_SEQ_LENGTH-2]


def dataset_reader(train=True, elmo=True):
    if elmo:
        elmo_token_indexer = ELMoTokenCharactersIndexer()
        reader = TransferLearnDatasetReader(
            tokenizer=spacy_tokenizer,
            token_indexers={"tokens": elmo_token_indexer}
        )
    else:
        reader = TransferLearnDatasetReader(
            tokenizer=bert_tokenizer,
            token_indexers={"tokens": bert_indexer()}
        )

    if train:
        train_dataset = reader.read(TRAIN_PATH)
        dev_dataset = reader.read(DEV_PATH)
        return train_dataset, dev_dataset
    else:
        test_dataset = reader.read(TEST_PATH)
        return test_dataset


def data_iterator(vocabulary):
    iterator = BucketIterator(batch_size=BATCH_SIZE,
                              sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocabulary)
    return iterator
