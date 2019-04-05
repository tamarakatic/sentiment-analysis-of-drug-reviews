import os

from data.definitions import OPTION_FILE, WEIGHT_FILE, PRETRAINED_ELMO
from elmo.prepare_elmo_data import data_iterator, dataset_reader
from elmo.base_model import BaseModel

import torch

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary

HIDDEN_DIM = 128
SEED = 42
LEARNING_RATE = 3e-4
EPOCHS = 6
BATCH_SIZE = 32


def main():
    cuda_gpu = torch.cuda.is_available()
    torch.manual_seed(SEED)

    elmo_embedders = ElmoTokenEmbedder(OPTION_FILE, WEIGHT_FILE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedders})

    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(),
                      HIDDEN_DIM,
                      bidirectional=True,
                      batch_first=True)
    )

    train_dataset, dev_dataset = dataset_reader()
    vocabulary = Vocabulary()

    model = BaseModel(word_embeddings=word_embeddings,
                      encoder=encoder,
                      vocabulary=vocabulary)

    model.cuda() if cuda_gpu else model

    iterator = data_iterator(vocabulary)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=dev_dataset,
        cuda_device=0 if cuda_gpu else -1,
        num_epochs=EPOCHS,
        patience=3
    )

    trainer.train()

    print("*******Save Model*******\n")

    output_elmo_model_file = os.path.join(PRETRAINED_ELMO, "lstm_elmo_model.bin")
    torch.save(model.state_dict(), output_elmo_model_file)


if __name__ == "__main__":
    main()
