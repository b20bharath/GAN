from base.dataloader import BaseDataLoader
from .mnist import MnistDataLoader, FashionMnistDataLoader, MixedMnistDataLoader


class DataLoaderFactory(object):
    __dict = {
        # Single Channel Image Datasets
        'mnist': MnistDataLoader,
        'fashion': FashionMnistDataLoader,
        'mixed_mnist': MixedMnistDataLoader,
    }

    @classmethod
    def get_dataloader(cls, name, input_size=1, latent_size=1, *args, **kwargs):
        # type: (str, int, int, *tuple, **dict) -> BaseDataLoader
        DL = cls.__dict[name]
        return DL(input_size, latent_size, *args, **kwargs)
