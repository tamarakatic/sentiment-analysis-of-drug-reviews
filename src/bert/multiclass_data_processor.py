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
        labels = [0, 1, 2]
        return labels

    def _create_examples(self, df):
        examples = []
        for (i, row) in df.iterrows():
            if i == 0:
                continue
            guid = row["Index"]
            text_a = row["review"]
            labels = row["rating"]

            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))

        return examples
