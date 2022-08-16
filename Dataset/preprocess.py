import os, pickle

from Dataset.data_loader_text import data_loader_text
from Dataset.tokenizer import tokenizer_spacy, tokenizer_nltk


def preprocess(tokenizer_name, dataset_loc, batch_size, train_ratio=0.6, valid_ratio=0.2, num_workers=0, shuffle=True, pin_memory=True):
    dataset_info = [tokenizer_name, dataset_loc, batch_size, train_ratio, valid_ratio, num_workers, shuffle, pin_memory]
    file = open("Dataset//cache//cache_info", "rb")
    dataset_info_cache = pickle.load(file)
    info_changed = (dataset_info_cache != dataset_info)

    if not os.listdir("Dataset//cache") or info_changed:
        tokenizer = None
        if tokenizer_name == "nltk":
            tokenizer = tokenizer_nltk()
        if tokenizer_name == "spacy":
            tokenizer = tokenizer_spacy()

        special_tokens = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        loader = data_loader_text(tokenizer.tokenize, special_tokens)
        loader.do_all(dataset_loc, batch_size, num_workers, shuffle, pin_memory, train_ratio, valid_ratio)

        file = open("Dataset//cache//cache", "wb")
        pickle.dump(loader, file)
        file.close()

        file = open("Dataset//cache//cache_info", "wb")
        pickle.dump(dataset_info, file)
        file.close()

    else:
        file = open("Dataset//cache//cache", "rb")
        loader = pickle.load(file)
        file.close()

    return loader.train_loader, loader.valid_loader, loader.eval_loader, loader.vocab_size


