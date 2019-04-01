import pandas as pd

from data.definitions import DEV_BERT_PATH, TRAIN_BERT_PATH, TEST_BERT_PATH
from data.definitions import OPTION_FILE, WEIGHT_FILE
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.module.token_embedders import ElmoTokenEmbedder
from allennlp.modules.elmo import batch_to_ids

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


def main():
    elmo_embedder = ElmoEmbedder(OPTION_FILE, WEIGHT_FILE)
    df_train = pd.read_csv(TRAIN_BERT_PATH)
    df_dev = pd.read_csv(DEV_BERT_PATH)


if __name__ == "__main__":
    main()
