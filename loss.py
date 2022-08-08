import torch.nn


def cross_entropy(weight=None, size_average=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    return torch.nn.CrossEntropyLoss(weight, size_average, ignore_index, reduction, label_smoothing)
