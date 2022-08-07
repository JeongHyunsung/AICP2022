from torch.nn.utils.rnn import pad_sequence


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        source = []
        for items in batch:
            for item in items[0]:
                source.append(item)
        target = [item[1] for item in batch]
        source = pad_sequence(source, batch_first=True, padding_value=self.pad_idx)
        target = pad_sequence(target, batch_first=True, padding_value=self.pad_idx)

        source = source.view([len(batch), 11, -1])

        return source, target
