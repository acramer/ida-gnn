import os
import dgl
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
from sklearn.datasets import make_classification
from functools import reduce

path = os.path.join

"""This script implements the functions for reading data.
"""
def create_cora_graphs(save=False):
    import dgl.data
    
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    
    u, v = g.edges()
    
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    
    train_g = dgl.remove_edges(g, eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    
    if save:
        with open('train_g.bin', 'wb')     as f: pkl.dump(train_g,     f)
        with open('train_pos_g.bin', 'wb') as f: pkl.dump(train_pos_g, f)
        with open('train_neg_g.bin', 'wb') as f: pkl.dump(train_neg_g, f)
        with open('test_pos_g.bin', 'wb')  as f: pkl.dump(test_pos_g,  f)
        with open('test_neg_g.bin', 'wb')  as f: pkl.dump(test_neg_g,  f)
    
    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

def create_ida_graphs(date='08/27/2021',threshold=0.07,date_offset=0,test=False,save=False):
    xndf = pd.read_csv('data/X_nodes.csv')
    xedf = pd.read_csv('data/X_edges.csv')
    yedf = pd.read_csv('data/Y_edges.csv')

    xndf = xndf[xndf['date']==date]
    yedf = yedf[(yedf['date']==date)&(yedf['idx']>=threshold)][['src','dst']]

    num_nodes = 64

    x = dgl.graph((xedf.values[:,0],xedf.values[:,1]),num_nodes=64) 
    x.ndata['feat'] = torch.Tensor(xndf[['precipitation','wind_gust','elevation']].values)
    g = dgl.graph((yedf.values[:,0],yedf.values[:,1]))
    u, v = g.edges()
    
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    
    num_nodes = x.number_of_nodes()
    self_edges = list(range(num_nodes))
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)+num_nodes), (list(u.numpy())+self_edges, list(v.numpy())+self_edges)))
    adj_neg = 1 - adj.todense()
    neg_u, neg_v = np.where(adj_neg != 0)
    
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    
    train_g = x
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=64)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=64)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=64)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=64)
    
    if save:
        with open('data/ida_train_g.bin', 'wb')     as f: pkl.dump(train_g,     f)
        with open('data/ida_train_pos_g.bin', 'wb') as f: pkl.dump(train_pos_g, f)
        with open('data/ida_train_neg_g.bin', 'wb') as f: pkl.dump(train_neg_g, f)
        with open('data/ida_test_pos_g.bin', 'wb')  as f: pkl.dump(test_pos_g,  f)
        with open('data/ida_test_neg_g.bin', 'wb')  as f: pkl.dump(test_neg_g,  f)

    if test:
        return train_g, test_pos_g, test_neg_g
    return train_g, train_pos_g, train_neg_g

# def load_cora_graphs():
def load_data(data_dir,test=False):
    with open(path(data_dir,'ida_train_g.bin'), 'rb') as f: train_g = pkl.load(f)
    if test:
        with open(path(data_dir,'ida_test_pos_g.bin'), 'rb')  as f: test_pos_g  = pkl.load(f)
        with open(path(data_dir,'ida_test_neg_g.bin'), 'rb')  as f: test_neg_g  = pkl.load(f)
        return train_g, test_pos_g, test_neg_g

    with open(path(data_dir,'ida_train_pos_g.bin'), 'rb') as f: train_pos_g = pkl.load(f)
    with open(path(data_dir,'ida_train_neg_g.bin'), 'rb') as f: train_neg_g = pkl.load(f)
    return train_g, train_pos_g, train_neg_g


# def load_data(data_dir):
#     """Load the CIFAR-10 dataset.
# 
#     Args:
#         data_dir: A string. The directory where data batches
#             are stored.
# 
#     Returns:
#         x_train: An numpy array of shape [50000, 3072].
#             (dtype=np.float32)
#         y_train: An numpy array of shape [50000,].
#             (dtype=np.int32)
#         x_test: An numpy array of shape [10000, 3072].
#             (dtype=np.float32)
#         y_test: An numpy array of shape [10000,].
#             (dtype=np.int32)
#     """
# 
#     ### YOUR CODE HERE
# 
#     with open(os.path.join(data_dir, 'test_batch'), mode='rb') as f:
#         raw_data = pickle.load(f, encoding="bytes")
#     x_test = raw_data[b'data']
#     y_test = np.array(raw_data[b'labels'])
# 
#     x_train, y_train = [], []
#     for i in range(1,6):
#         with open(os.path.join(data_dir, 'data_batch_'+str(i)), mode='rb') as f:
#             raw_data = pickle.load(f, encoding="bytes")
#         x_train.append(raw_data[b'data'])
#         y_train.append(np.array(raw_data[b'labels']))
# 
#     x_train = np.concatenate(x_train,0).reshape(-1,3,32,32).transpose(0,2,3,1)
#     y_train = np.concatenate(x_train,0)
#     ### END CODE HERE
# 
#     return x_train, y_train, x_test, y_test


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


# from ImageUtils import ImageDataset
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
