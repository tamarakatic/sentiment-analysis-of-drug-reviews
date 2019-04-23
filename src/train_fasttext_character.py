from pathlib import Path

from data.definitions import DATA_PATH, PRETRAINED_FLAIR, FLAIR_LOSS, FLAIR_WEIGHTS

from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter


def main():
    train_dev_corpus = NLPTaskDataFetcher.load_classification_corpus(
        Path(DATA_PATH),
        train_file='flair_train.csv',
        test_file='flair_test.csv',
        dev_file='flair_dev.csv')

    label_dict = train_dev_corpus.make_label_dictionary()

    word_embeddings = [WordEmbeddings('crawl'),
                       FlairEmbeddings('news-forward-fast', chars_per_chunk=128),
                       FlairEmbeddings('news-backward-fast', chars_per_chunk=128)]

    document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                rnn_type='LSTM',
                                                hidden_size=128,
                                                reproject_words=True,
                                                reproject_words_dimension=64)

    classifier = TextClassifier(document_embeddings,
                                label_dictionary=label_dict,
                                multi_label=False)

    trainer = ModelTrainer(classifier, train_dev_corpus)
    trainer.train(PRETRAINED_FLAIR,
                  max_epochs=40,
                  learning_rate=0.2,
                  mini_batch_size=32,
                  embeddings_in_memory=False,
                  checkpoint=True)

    plotter = Plotter()
    plotter.plot_training_curves(FLAIR_LOSS)
    plotter.plot_weights(FLAIR_WEIGHTS)


if __name__ == "__main__":
    main()
