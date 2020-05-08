from __future__ import absolute_import
import torch as tr
import pickle
from base.dataloader import BaseDataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import numpy as np

def normalize_mnist_images(x):
    x = x[:, None, :, :]
    return 2 * (x.type(tr.float32) / 255.0) - 1.0


class MnistDataLoader(BaseDataLoader):
    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=True, *args, **kwargs):
        super(MnistDataLoader, self).__init__((28, 28), 100, train_batch_size, test_batch_size, get_tensor, supervised)

    def get_data(self):
        MNIST("../data/mnist", download=True)

        train_data, train_labels = tr.load('../data/mnist/MNIST/processed/training.pt')
        test_data, test_labels = tr.load('../data/mnist/MNIST/processed/test.pt')

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels

class CifarDataLoader(BaseDataLoader):
    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=True, *args, **kwargs):
        super(CifarDataLoader, self).__init__((3, 32, 32,), 100, train_batch_size, test_batch_size, get_tensor, supervised)
    
    def load_cfar10_batch(self,cifar10_dataset_folder_path, batch_id):
        if batch_id == 0:
            with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
                batch = pickle.load(file, encoding='latin1')
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 1, 2, 3)
                labels = batch['labels']
            return features, labels
        else:
            with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
                batch = pickle.load(file, encoding='latin1')
            
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 1, 2, 3)
                labels = batch['labels']
            
            return (features, labels)
    
    def get_data(self):
        CIFAR10("../data/cifar", download=True)
        train = []
        features_train ={}
        labels_train = {}
        
        for i in range(0,6):
            if i == 0:
                test_data, test_labels = self.load_cfar10_batch("../data/cifar/cifar-10-batches-py",i)
            else:
                train.append(self.load_cfar10_batch("../data/cifar/cifar-10-batches-py",i))
        for i in range(1,6):
            (features_train[i], labels_train[i]) = train[i-1]
        train_data = np.append(features_train[1],features_train[2], axis=0)
        for i in range(2,6):
            train_data = np.append(train_data,features_train[i], axis=0)
        train_labels = np.append(labels_train[1],labels_train[2], axis=0)
        for i in range(2,6):
            train_labels = np.append(train_labels,labels_train[i], axis=0)
        train_data = tr.from_numpy(train_data)
        train_labels = tr.from_numpy(train_labels)
        test_data = tr.from_numpy(np.array(test_data))
        test_labels = tr.from_numpy(np.array(test_labels))
        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels
    
    

class FashionMnistDataLoader(BaseDataLoader):
    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True):
        super(FashionMnistDataLoader, self).__init__((28, 28), 100, train_batch_size, test_batch_size, get_tensor,
                                                     supervised=True)

    def get_data(self):
        FashionMNIST('../data/fashion', download=True)
        train_data, train_labels = tr.load('../data/fashion/FashionMNIST/processed/training.pt')
        test_data, test_labels = tr.load('../data/fashion/FashionMNIST/processed/test.pt')

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels


class MixedMnistDataLoader(BaseDataLoader):
    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True):
        super(MixedMnistDataLoader, self).__init__((28, 28), None, train_batch_size, test_batch_size, get_tensor,
                                                   supervised=True)

    def get_data(self):
        FashionMNIST('../data/fashion', download=True)
        MNIST('../data/fashion', download=True)

        mnist_train_data, mnist_train_labels = tr.load('../data/mnist/processed/training.pt')
        mnist_test_data, mnist_test_labels = tr.load('../data/mnist/processed/test.pt')

        fmnist_train_data, fmnist_train_labels = tr.load('../data/fashion/processed/training.pt')
        fmnist_test_data, fmnist_test_labels = tr.load('../data/fashion/processed/test.pt')

        train_data = tr.cat([mnist_train_data, fmnist_train_data])
        train_labels = tr.cat([mnist_train_labels, 10 + fmnist_train_labels])
        test_data = tr.cat([mnist_test_data, fmnist_test_data])
        test_labels = tr.cat([mnist_test_labels, 10 + fmnist_test_labels])

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels
