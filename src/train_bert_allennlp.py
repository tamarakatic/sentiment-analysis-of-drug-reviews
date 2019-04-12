import os
import torch

from prepare_allennlp_data import data_iterator, dataset_reader
from tl_allennlp.base_model import BaseModel
from tl_allennlp.definitions import PRETRAINED_BERT
from tl_allennlp.bert_encoder import BertSentencePooler

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import Trainer

HIDDEN_DIM = 128
SEED = 42
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 30
BATCH_SIZE = 32


def main():
    cuda_device = -1

    torch.manual_seed(SEED)

    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True
    )
    word_embedding = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                            allow_unmatched_keys=True)

    train_dataset, dev_dataset = dataset_reader(train=True, elmo=False)
    vocab = Vocabulary()

    encoder = BertSentencePooler(vocab)

    model = BaseModel(word_embeddings=word_embedding,
                      encoder=encoder,
                      vocabulary=vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)

    iterator = data_iterator(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=dev_dataset,
        cuda_device=cuda_device,
        num_epochs=EPOCHS,
        patience=5
    )

    trainer.train()

    print("*******Save Model*******\n")

    output_bert_model_file = os.path.join(PRETRAINED_BERT, "bert_model.bin")
    torch.save(model.state_dict(), output_bert_model_file)


if __name__ == '__main__':
    main()
