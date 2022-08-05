import json, pandas as pd
import os
from torchtext.data import Field, BucketIterator


class DataLoader_text:
    def __init__(self, tokenizer, init_token, eos_token, unk_token):
        self.tokenizer = tokenizer
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        print("DataLoader_text initialized")

    def get_data(self, dataset_loc):
        filename_list = os.listdir(dataset_loc)
        print(filename_list)
        file_list = [file for file in filename_list if file.endswith(".json")]

        datasets = []
        for file in file_list:
            filename = dataset_loc + "//" + file
            with open(filename, 'r') as f:
                datasets.append(json.load(f))

        dataset_2darr = []
        for dataset in datasets:
            for data in dataset:
                data_1darr = []
                history_num = len(data["text_history"])
                for i in range(history_num):
                    data_1darr.append(data["text_history"][history_num - 1 - i])
                data_1darr.append(data["text"])
                data_1darr.append(data["text_next"])
                dataset_2darr.append(data_1darr)

        dataset_frame = pd.DataFrame(dataset_2darr, columns=['history-10', 'history-9', 'history-8', 'history-7',
                                                             'history-6', 'history-5', 'history-4', 'history-3',
                                                             'history-2', 'history-1', 'current', 'next'])
        return dataset_frame

    def build_vocab(self):
        pass

    def tokenization(self):
        pass

    def token2index(self):
        pass

    def dataloader(self):
        pass
