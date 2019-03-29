class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()
