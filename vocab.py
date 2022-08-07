from tqdm import tqdm


class Vocab:
    def __init__(self, freq_threshold, max_size, tokenizer, special_tokens):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.itos = special_tokens
        self.stoi = {i:j for j, i in self.itos.items()}
        self.is_built = False

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, train_dataset):
        frequencies = {}
        idx = len(self.special_tokens)

        for i in tqdm(range(train_dataset.shape[0]*train_dataset.shape[1])):
            sentence = train_dataset.iloc[i//train_dataset.shape[1], i % train_dataset.shape[1]]
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        frequencies = {i:j for i, j in frequencies.items() if j > self.freq_threshold}
        frequencies = dict(sorted(frequencies.items(), key=lambda x: -x[1])[:self.max_size-idx])

        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
        self.is_built = True

        return

    def to_index(self, sentence):
        if not self.is_built:
            raise Exception("Vocabulary not built")
        tokenized_sentence = self.tokenizer(sentence)
        indexed_sentence = []
        for token in tokenized_sentence:
            if token in self.stoi.keys():
                indexed_sentence.append(self.stoi[token])
            else:
                indexed_sentence.append(self.stoi['<UNK>'])
        return indexed_sentence





