import json, pandas as pd, numpy as np
import os, random
from torchtext.data import Field, BucketIterator


class data_loader_text:
    def __init__(self):
        self.dataset_frame = None
        self.train_frame = None
        self.valid_frame = None
        self.eval_frame = None
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
        self.dataset_frame = dataset_frame
        return

    def split_dataset(self, train_ratio=0.6, valid_ratio=0.2):
        dataset_length = len(self.dataset_frame)
        valid_split_idx = int(dataset_length*train_ratio)
        eval_split_idx = int(dataset_length*(train_ratio+valid_ratio))
        data_idx = list(range(dataset_length))
        np.random.shuffle(data_idx)

        train_idx, valid_idx, eval_idx = data_idx[:valid_split_idx], data_idx[valid_split_idx:eval_split_idx], data_idx[eval_split_idx:]
        self.train_frame = self.dataset_frame.iloc[train_idx]
        self.valid_frame = self.dataset_frame.iloc[valid_idx]
        self.eval_frame = self.dataset_frame.iloc[valid_idx]
        print(self.train_frame, self.valid_frame, self.eval_frame)
        return

    def build_vocab(self):
        pass

    def tokenization(self, tokenizer, init_token, eos_token, unk_token):
        pass

    def token2index(self):
        pass

    def dataloader(self):
        pass
