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

from random import sample
class DL:
    def __init__(self,*args):
        self.d = np.array(list(zip(*args)))
        self.len = len(args[0])

    def __len__(self):
        return self.len

    def __iter__(self):
        i = sample(range(self.len),self.len)
        return iter(self.d[i])

def create_ida_graphs(threshold=0.05,window_size=9,test=False,save=False):
    # assert date in DATES and DATE2NUM[date] + window_size <= len(DATES)
    xndf = pd.read_csv('data/X_nodes.csv')
    xedf = pd.read_csv('data/X_edges.csv')
    yedf = pd.read_csv('data/Y_edges.csv')

    num_nodes = 64

    # X,Y_train,Z_train,Y_test,Z_test = [],[],[],[],[]
    X,Y,Z = [],[],[]
    # for d in range(len(DATES)+1-window_size):
    #     date = NUM2DATE[d+(window_size-1)//2]
    for date in ['08/28/2021','08/27/2021','08/29/2021']:
        d = DATE2NUM[date] - (window_size-1)//2
        y = yedf[(yedf['date']==date)&(yedf['idx']>=threshold)][['src','dst']]

        x = dgl.graph((xedf.values[:,0],xedf.values[:,1]),num_nodes=64) 
        node_data  = [torch.Tensor(xndf[xndf['date']==date][['elevation']].values)]
        node_data += [torch.Tensor(xndf[xndf['date']==NUM2DATE[d+i]][['precipitation','wind_gust']].values) for i in range(window_size)]
        # node_data += [torch.Tensor(xndf[xndf['date']==NUM2DATE[d]][['precipitation','wind_gust']].values) for d in range(DATE2NUM[date],DATE2NUM[date]+window_size)]
        x.ndata['feat'] = torch.cat(node_data,dim=1)
        g = dgl.graph((y.values[:,0],y.values[:,1]), num_nodes=64)

        X.append(x)
        Y.append(g)

        u, v = g.edges()
        num_nodes = x.number_of_nodes()
        self_edges = list(range(num_nodes))
        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)+num_nodes), (list(u.numpy())+self_edges, list(v.numpy())+self_edges)))
        adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
        neg_u, neg_v = neg_u[neg_eids], neg_v[neg_eids]
        
        Z.append(dgl.graph((neg_u,neg_v), num_nodes=64))

        # u, v = g.edges()
        # 
        # eids = np.arange(g.number_of_edges())
        # eids = np.random.permutation(eids)
        # test_size = int(len(eids) * 0.1)
        # train_size = g.number_of_edges() - test_size
        # test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        # train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
        # 
        # num_nodes = x.number_of_nodes()
        # self_edges = list(range(num_nodes))
        # # Find all negative edges and split them for training and testing
        # adj = sp.coo_matrix((np.ones(len(u)+num_nodes), (list(u.numpy())+self_edges, list(v.numpy())+self_edges)))
        # adj_neg = 1 - adj.todense()
        # neg_u, neg_v = np.where(adj_neg != 0)
        # 
        # neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
        # test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        # train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
        # 
        # train_g = x
        # train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=64)
        # train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=64)
        # test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=64)
        # test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=64)
        # 
        # X.append(train_g)
        # Y_train.append(train_pos_g)
        # Z_train.append(train_neg_g)
        # Y_test.append(test_pos_g)
        # Z_test.append(test_neg_g)
    test_data = DL(X[:1],Y[:1],Z[:1])
    train_data = DL(X[1:],Y[1:],Z[1:])
    return train_data, test_data
    

DATES = ['08/22/2021',
         '08/23/2021',
         '08/24/2021',
         '08/25/2021',
         '08/26/2021',#
         '08/27/2021',#
         '08/28/2021', 
         '08/29/2021',# 
         '08/30/2021',
         '08/31/2021', 
         '09/01/2021',# 
         '09/02/2021',# 
         # '09/03/2021',
         # '09/04/2021',
         # '09/05/2021',
         # '09/06/2021',
         # '09/07/2021',
         # '09/08/2021',
         # '09/09/2021',
         # '09/10/2021',
         # '09/11/2021',
         # '09/12/2021',
         # '09/13/2021',
         # '09/14/2021',
         # '09/15/2021',
         # '09/16/2021',
         # '09/17/2021',
         # '09/18/2021',
         # '09/19/2021',
         # '09/20/2021',
         # '09/21/2021',
         # '09/22/2021',
         # '09/23/2021',
         # '09/24/2021',
         # '09/25/2021',
         # '09/26/2021',
         # '09/27/2021',
        ]
DATE2NUM = dict(zip(DATES,range(len(DATES))))
NUM2DATE = dict(zip(range(len(DATES)),DATES))

