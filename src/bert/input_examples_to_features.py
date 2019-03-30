from .input_feature import InputFeatures


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    classifier = "[CLS]"
    separator = "[SEP]"

    for (idx, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > (max_seq_length - 2):  # 2 for [CLS] and [SEP]
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = [classifier] + tokens_a + [separator]
        segment_ids = [0] * len(tokens)  # 0 for text_a, 1 for text_b

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # mask has 1 for real token and 0 for padding
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_ids = [0.0 if idx != example.labels else 1.0]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_ids))

    return features
