import os
import pickle
import numpy as np
from sklearn.datasets import make_classification
from functools import reduce

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE

    with open(os.path.join(data_dir, 'test_batch'), mode='rb') as f:
        raw_data = pickle.load(f, encoding="bytes")
    x_test = raw_data[b'data']
    y_test = np.array(raw_data[b'labels'])

    x_train, y_train = [], []
    for i in range(1,6):
        with open(os.path.join(data_dir, 'data_batch_'+str(i)), mode='rb') as f:
            raw_data = pickle.load(f, encoding="bytes")
        x_train.append(raw_data[b'data'])
        y_train.append(np.array(raw_data[b'labels']))

    x_train = np.concatenate(x_train,0).reshape(-1,3,32,32).transpose(0,2,3,1)
    y_train = np.concatenate(x_train,0)
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir, data_file='private_test_images.npy'):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE

    if len(data_dir) > 4 and data_dir[-4:] == '.npy':
        test_file = data_dir
    else:
        test_file = os.path.join(data_dir, data_file)
    x_test = np.load(test_file)

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE

    split_index=int(len(x_train)*train_ratio)

    x_train_new = x_train[:split_index].copy()
    y_train_new = y_train[:split_index].copy()
    x_valid = x_train[split_index:].copy()
    y_valid = y_train[split_index:].copy()

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid


from ImageUtils import ImageDataset
from torch.utils.data import random_split, DataLoader
def parse_data(x_data, label_data, batch_size=128, train_ratio=0.8, **kwargs):
    dataset = ImageDataset(x_data, label_data, **kwargs)
    if train_ratio < 1 and train_ratio > 0: 
        train_size = int(len(x_data)*train_ratio)
        valid_size = len(x_data)-train_size
        train, valid = random_split(dataset, [train_size, valid_size])
        return DataLoader(train, num_workers=2, batch_size=batch_size, shuffle=True, drop_last=False), DataLoader(valid, num_workers=2, batch_size=batch_size, shuffle=False, drop_last=False)
    return DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=False)


def load_fake_data(fnum=10,fclass=4,fshape=(32,32,3)):
    total_features = reduce(lambda x,y:x*y,fshape)
    informative = min(50, total_features)
    redundant = total_features-informative
    fx, fy = make_classification(
                n_features=total_features,
                n_informative=informative,
                n_redundant=redundant,
                n_repeated=0,
                n_samples=fnum,
                n_classes=fclass,
                n_clusters_per_class=1,
                class_sep=2,
                )
    fx -= np.min(fx)
    fx *= 255/np.max(fx)
    fx = np.round(fx)
    fx = fx.astype('uint8')
    return fx, fy
