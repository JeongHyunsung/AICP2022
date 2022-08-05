class Vocab:
    def __init__(self, freq_threshold, max_size, tokenizer, special_tokens):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.tokenizer = tokenizer
        self.itos = special_tokens
        self.stoi = {i:j for j, i in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, dataset):
        pass

    def to_index(self, sentence):
        pass





