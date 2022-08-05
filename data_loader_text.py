import json, os, pandas as pd

import spacy, torchtext
from t2i import T2I


def get_text_data(filename):
    with open(filename, 'r') as f:
        dataset = json.load(f)

    dataset_2darr = []

    for data in dataset:
        data_1darr = []
        history_num = len(data["text_history"])
        for i in range(history_num):
            data_1darr.append(data["text_history"][history_num-1-i])
        data_1darr.append(data["text"])
        data_1darr.append(data["text_next"])

        dataset_2darr.append(data_1darr)

    dataset_frame = pd.DataFrame(dataset_2darr, columns=['history-10', 'history-9', 'history-8', 'history-7',
                                                         'history-6', 'history-5', 'history-4', 'history-3',
                                                         'history-2', 'history-1', 'current', 'next'])
    print(dataset_frame.shape)
    return dataset_frame


def tokenization(dataset):
    nlp = spacy.load("en_core_web_sm")
    print(nlp.vocab)
    tokenized_1darr = []

    print(dataset.shape[0], dataset.shape[1])
    tokenized_2darr = []

    for i in range(dataset.shape[0]):
        tokenized_1darr = []
        for j in range(dataset.shape[1]):
            tokenized_1darr.append([token.text for token in nlp(dataset.iloc[i, j])])
        tokenized_2darr.append(tokenized_1darr)

    tokenized_frame = pd.DataFrame(tokenized_2darr, columns=['history-10', 'history-9', 'history-8', 'history-7',
                                                             'history-6', 'history-5', 'history-4', 'history-3',
                                                             'history-2', 'history-1', 'current', 'next'])
    return tokenized_frame



def token2index(dataset):

    pass


if __name__ == "__main__":
    print(tokenization(get_text_data("MovieChat//data//12 Years a Slave.json")))
