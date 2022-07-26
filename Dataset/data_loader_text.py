import os, json, pandas as pd, numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from Dataset.vocab import Vocab
from Dataset.dataset import Dataset
from Dataset.utils import MyCollate



class data_loader_text:
    def __init__(self, tokenizer, special_tokens):
        self.special_tokens = special_tokens  # dictionary of special tokens <eos>, <sos>, <unk>, <pad>...
        self.tokenizer = tokenizer  # tokenizer function split sentence to small tokens
        self.dataset_frame = None  # dataframe which include all data
        self.random_data_idx = None
        self.train_frame = None
        self.valid_frame = None
        self.eval_frame = None
        self.train_dataset = None  # train dataset
        self.valid_dataset = None  # valid dataset
        self.eval_dataset = None  # eval dataset
        self.vocab = None  # Vocab class
        self.vocab_size = None  # number of total tokens in vocabulary
        self.train_loader = None  # dataloader using train dataset
        self.valid_loader = None  # dataloader using valid dataset
        self.eval_loader = None  # dataloader using eval dataset

    def get_json(self, dataset_loc):  # load all json files into dictionary, extract text data into dataframe
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
                if data["text_next_exist"] == 0:
                    continue
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

    def split_dataframe(self, train_ratio=0.6, valid_ratio=0.2):  # split dataframe randomly
        print("\nSplit dataframe into train/valid/eval ...")
        if self.dataset_frame is None:
            raise Exception("error : Dataset not found")
        dataset_length = len(self.dataset_frame)
        valid_split_idx = int(dataset_length*train_ratio)
        eval_split_idx = int(dataset_length*(train_ratio+valid_ratio))
        data_idx = list(range(dataset_length))
        np.random.shuffle(data_idx)
        self.random_data_idx = data_idx

        train_idx, valid_idx, eval_idx = data_idx[:valid_split_idx], data_idx[valid_split_idx:eval_split_idx], data_idx[eval_split_idx:]
        self.train_frame = self.dataset_frame.iloc[train_idx]
        self.valid_frame = self.dataset_frame.iloc[valid_idx]
        self.eval_frame = self.dataset_frame.iloc[valid_idx]
        self.train_frame = self.train_frame.reset_index(drop=True)
        self.valid_frame = self.valid_frame.reset_index(drop=True)
        self.eval_frame = self.eval_frame.reset_index(drop=True)
        return

    def make_vocab(self, vocab_freq_threshold, vocab_max_size):  # using train data, make vocabulary correspond to tokenizer
        print("\nMake vocabulary from train dataframe ...")
        if self.train_frame is None:
            raise Exception("error : Train dataset not found")
        vocab = Vocab(vocab_freq_threshold, vocab_max_size, self.tokenizer, self.special_tokens)
        vocab.build_vocab(self.train_frame)

        self.vocab = vocab
        self.vocab_size = len(vocab)
        return

    def make_dataset(self):
        print("\nMake dataset ...")
        if self.vocab is None:
            raise Exception("error : Vocabulary not found")
        self.train_dataset = Dataset(self.train_frame,
                                     self.vocab,
                                     ['history-10', 'history-9', 'history-8', 'history-7',
                                      'history-6', 'history-5', 'history-4', 'history-3',
                                      'history-2', 'history-1', 'current'],
                                      ['next'])
        self.valid_dataset = Dataset(self.valid_frame,
                                     self.vocab,
                                     ['history-10', 'history-9', 'history-8', 'history-7',
                                      'history-6', 'history-5', 'history-4', 'history-3',
                                      'history-2', 'history-1', 'current'],
                                     ['next'])
        self.eval_dataset = Dataset(self.eval_frame,
                                    self.vocab,
                                    ['history-10', 'history-9', 'history-8', 'history-7',
                                     'history-6', 'history-5', 'history-4', 'history-3',
                                     'history-2', 'history-1', 'current'],
                                    ['next'])
        return

    def set_train_loader(self, batch_size, num_workers=0, shuffle=True, pin_memory=True):
        print("\nSet train data loader ...")
        if self.train_dataset is None:
            raise Exception("error : Train dataset not found")
        pad_idx = self.vocab.stoi["<PAD>"]
        loader = DataLoader(self.train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=MyCollate(pad_idx=pad_idx))
        self.train_loader = loader
        return

    def set_valid_loader(self, batch_size, num_workers=0, shuffle=True, pin_memory=True):
        print("\nSet valid data loader ...")
        pad_idx = self.vocab.stoi["<PAD>"]
        loader = DataLoader(self.valid_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=MyCollate(pad_idx=pad_idx))
        self.valid_loader = loader
        return

    def set_eval_loader(self, batch_size, num_workers=0, shuffle=True, pin_memory=True):
        print("\nSet eval data loader ...")
        pad_idx = self.vocab.stoi["<PAD>"]
        loader = DataLoader(self.eval_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            collate_fn=MyCollate(pad_idx=pad_idx))
        self.eval_loader = loader
        return

    def set_loader(self, batch_size, num_workers=0, shuffle=True, pin_memory=True):
        self.set_train_loader(batch_size, num_workers, shuffle, pin_memory)
        self.set_valid_loader(batch_size, num_workers, shuffle, pin_memory)
        self.set_eval_loader(batch_size, num_workers, shuffle, pin_memory)

    def do_all(self, vocab_freq_threshold, vocab_max_size, dataset_loc, batch_size, num_workers=0, shuffle=True, pin_memory=True, train_ratio=0.6, valid_ratio=0.2):
        self.get_json(dataset_loc)
        self.split_dataframe(train_ratio, valid_ratio)
        self.make_vocab(vocab_freq_threshold, vocab_max_size)
        self.make_dataset()
        self.set_loader(batch_size, num_workers, shuffle, pin_memory)
        return
