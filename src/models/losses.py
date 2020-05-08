import torch as tr
from torch import distributions as dist
from torch.nn import functional as F
from torch.nn.modules import loss


def sigmoid_cross_entropy_loss(logits, labels):
    if labels == 0.:
        labels = tr.zeros_like(logits)
    elif labels == 1.:
        labels = tr.ones_like(logits)

    losses = tr.max(logits, tr.zeros_like(logits)) - logits * labels + tr.log(1 + tr.exp(-tr.abs(logits)))
    return losses.mean()


def l2_reg(params, l=0.0002):
    loss = 0.
    for param in params:
        loss += l * tr.sum(param ** 2)
    return loss

