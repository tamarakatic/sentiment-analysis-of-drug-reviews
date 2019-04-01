import pandas as pd

from .input_example import InputExample
from .data_processor import DataProcessor
from data.definitions import TRAIN_BERT_PATH, TEST_BERT_PATH, DEV_BERT_PATH


class MultiLabelDataProcessor(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def get_train_examples(self, data_dir):
        df_train = pd.read_csv(TRAIN_BERT_PATH)
        df_train.dropna(inplace=True)
        return self.__create_examples(df_train)

    def get_dev_examples(self, data_dir):
        df_dev = pd.read_csv(DEV_BERT_PATH)
        df_dev.dropna(inplace=True)
        return self.__create_examples(df_dev)

    def get_test_examples(self, data_dir, data_file_name):
        df_test = pd.read_csv(TEST_BERT_PATH)
        df_test.dropna(inplace=True)
        return self.__create_examples(df_test)

    def get_labels(self):
        if self.labels is None:
            self.labels = [0, 1, 2]
        return self.labels

    def __create_examples(self, df):
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[3]
            labels = row[4]

            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))

        return examples
