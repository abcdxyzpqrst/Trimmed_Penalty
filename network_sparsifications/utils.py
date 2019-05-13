import numpy as np
import torch
from torch import nn

def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    f = fts(dummy_input)
    return int(np.prod(f.size()[1:]))

def conv_norm(weights):
    """
    For 4-D tensor weights, compute the elementwise norm
    weights have shape [out_channels, in_channels, kernel_height, kernel_width]
    """
    return torch.sqrt((weights**2).sum(dim=3).sum(dim=2).sum(dim=1))

def accuracy(model, valid_loader):
    correct = 0.0
    for images, labels in valid_loader:
        images = images.view(-1, 28*28)
        if torch.cuda.is_available():
            images, labels = images.cuda(async=True), labels.cuda(async=True)

        output = model(images)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
    return 100. * correct / len(valid_loader.dataset)
