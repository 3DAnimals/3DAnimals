import torch.nn as nn


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    else:
        raise NotImplementedError
