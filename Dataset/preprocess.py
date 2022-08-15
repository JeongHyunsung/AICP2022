import os, pickle

from Dataset.data_loader_text import data_loader_text
from Dataset.tokenizer import tokenizer_spacy, tokenizer_nltk


def preprocess(batch_size):
    if not os.listdir("Dataset//cache"):
        tokenizer = tokenizer_nltk()
        special_tokens = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        loader = data_loader_text(tokenizer.tokenize, special_tokens)
        loader.do_all("MovieChat//data", batch_size)

        file = open("Dataset//cache//cache", "wb")
        pickle.dump(loader, file)
        file.close()

    else:
        file = open("Dataset//cache//cache", "rb")
        loader = pickle.load(file)
        file.close()

    return loader.train_loader, loader.valid_loader, loader.eval_loader, loader.vocab_size


