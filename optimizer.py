import torch.optim


def sgd(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False, foreach=None):
    return torch.optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)




