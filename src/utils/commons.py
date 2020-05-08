import torch as tr
from torch import nn
from torch.nn import functional as F

from configs import Config


class NLinear(nn.Sequential):
    def __init__(self, in_features, units, act=nn.ELU):
        """

        :type units: list of other features
        """
        layers = [nn.Linear(in_features, units[0])]
        for i in range(len(units) - 1):
            in_features, out_features = units[i:i + 2]
            layers.append(act())
            layers.append(nn.Linear(in_features, out_features))

        super(NLinear, self).__init__(*layers)
        if Config.use_gpu:
            self.cuda()

class ConvBlock(nn.Sequential):
    def __init__(self, in_filters, out_filters, bn=True, kernel_size=3, stride=2):
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]
        if bn:
            layers.append(nn.BatchNorm2d(out_filters, 0.8))
        layers.append(nn.LeakyReLU(0.3, inplace=True))

        super(ConvBlock, self).__init__(*layers)


class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, bn=True, kernel_size=5, stride=2, output_padding=(0, 0), padding=(2, 2)):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,
                               padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        super(UpConvBlock, self).__init__(*layers)


def normalize_(x):
    # min_value, _ = tr.min(x, dim=-1, keepdim=True)
    # max_value, _ = tr.max(x, dim=-1, keepdim=True)
    # return 2 * (x - min_value) / (max_value - min_value) - 1

    return x
