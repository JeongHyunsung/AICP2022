import torch
from torch.utils.data import Dataset


class Train_Dataset(Dataset):
    def __init__(self, frame, vocab, source_column, target_column):
        self.frame = frame
        self.vocab = vocab
        self.source = self.frame[source_column]
        self.target = self.frame[target_column]



    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        source_item = self.source.iloc[index]
        target_item = self.target.iloc[index]

        # in this task
        assert source_item.shape[0] == 11
        assert target_item.shape[0] == 1

        source_indexed_item = []
        source_indexed_item_max_length = 0
        for i in range(source_item.shape[0]):
            indexed_sentence = [self.vocab.stoi["<SOS>"]]
            indexed_sentence += self.vocab.to_index(source_item.iloc[i])
            indexed_sentence.append(self.vocab.stoi["<EOS>"])
            source_indexed_item.append(indexed_sentence)
            if len(indexed_sentence) > source_indexed_item_max_length:
                source_indexed_item_max_length = len(indexed_sentence)
        for i in range(source_item.shape[0]):
            indexed_sentence_length = len(source_indexed_item[i])
            for j in range(source_indexed_item_max_length-indexed_sentence_length):
                source_indexed_item[i].append(self.vocab.stoi["<PAD>"])

        target_indexed_item = [self.vocab.stoi["<SOS>"]]
        target_indexed_item += self.vocab.to_index(target_item.iloc[0])
        target_indexed_item.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(source_indexed_item), torch.tensor(target_indexed_item)


class Valid_Dataset(Dataset):
    def __init__(self, frame, vocab, source_column, target_column):
        self.frame = frame
        self.vocab = vocab
        self.source = self.frame[source_column]
        self.target = self.frame[target_column]


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        pass
