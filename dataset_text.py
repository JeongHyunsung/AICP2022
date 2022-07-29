import json, os, pandas as pd


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
    print(dataset_frame)


def tokenization(dataset):
    pass


def token2index(dataset):
    pass


if __name__ == "__main__":
    get_text_data("MovieChat//data//12 Years a Slave.json")
