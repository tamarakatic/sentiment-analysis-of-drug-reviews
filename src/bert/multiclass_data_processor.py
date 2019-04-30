import pandas as pd

from .input_example import InputExample
from .data_processor import DataProcessor


class MultiClassDataProcessor(DataProcessor):

    def get_train_examples(self, data_path):
        df_train = pd.read_csv(data_path)
        df_train.dropna(inplace=True)
        return self._create_examples(df_train)

    def get_dev_examples(self, data_path):
        df_dev = pd.read_csv(data_path)
        df_dev.dropna(inplace=True)
        return self._create_examples(df_dev)

    def get_test_examples(self, data_path):
        df_test = pd.read_csv(data_path)
        df_test.dropna(inplace=True)
        return self._create_examples(df_test)

    def get_labels(self):
        if self.labels is None:
            self.labels = [0, 1, 2]
        return self.labels

    def _create_examples(self, df):
        examples = []
        for (i, row) in enumerate(df):
            if i == 0:
                continue
            guid = row[0]
            text_a = row[3]
            labels = row[4]

            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))

        return examples
