import os, json, pandas as pd, numpy as np
from vocab import Vocab

from tqdm import tqdm
from dataset import Train_Dataset, Valid_Dataset


class data_loader_text:
    def __init__(self, tokenizer, special_tokens):
        self.special_tokens = special_tokens  # dictionary of special tokens <eos>, <sos>, <unk>, <pad>...
        self.tokenizer = tokenizer  # tokenizer function split sentence to small tokens, (한국어 이용한다면 tokenizer 교체)
        self.dataset_frame = None  # dataset
        self.train_frame = None
        self.valid_frame = None
        self.eval_frame = None
        self.train_dataset = None
        self.valid_dataset = None
        self.eval_dataset = None
        self.vocab = None  # Vocab class

    def get_json(self, dataset_loc):
        filename_list = os.listdir(dataset_loc)
        file_list = [file for file in filename_list if file.endswith(".json")]

        datasets = []
        print("\nLoad json file ...")
        for file in tqdm(file_list):
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

    def split_dataframe(self, train_ratio=0.6, valid_ratio=0.2):
        print("\nSplit dataframe into train/valid/eval ...")
        if self.dataset_frame is None:
            raise Exception("Dataset not found")
        dataset_length = len(self.dataset_frame)
        valid_split_idx = int(dataset_length*train_ratio)
        eval_split_idx = int(dataset_length*(train_ratio+valid_ratio))
        data_idx = list(range(dataset_length))
        np.random.shuffle(data_idx)

        train_idx, valid_idx, eval_idx = data_idx[:valid_split_idx], data_idx[valid_split_idx:eval_split_idx], data_idx[eval_split_idx:]
        self.train_frame = self.dataset_frame.iloc[train_idx]
        self.valid_frame = self.dataset_frame.iloc[valid_idx]
        self.eval_frame = self.dataset_frame.iloc[valid_idx]
        self.train_frame = self.train_frame.reset_index(drop=True)
        self.valid_frame = self.valid_frame.reset_index(drop=True)
        self.eval_frame = self.eval_frame.reset_index(drop=True)
        return

    def make_vocab(self):
        print("\nMake vocabulary from train dataframe ...")
        if self.train_frame is None:
            raise Exception("Train dataset not found")
        vocab = Vocab(1, 50000, self.tokenizer, self.special_tokens)
        vocab.build_vocab(self.train_frame)

        self.vocab = vocab
        return

    def make_dataset(self):
        print("\nMake dataset ...")
        if self.vocab is None:
            raise Exception("Vocabulary not found")
        self.train_dataset = Train_Dataset(self.train_frame,
                                           self.vocab,
                                           ['history-10', 'history-9', 'history-8', 'history-7',
                                            'history-6', 'history-5', 'history-4', 'history-3',
                                            'history-2', 'history-1', 'current'],
                                           ['next'])
        return

    def token2index(self):
        pass

    def dataloader(self):
        pass
