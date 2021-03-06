import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import dgl.function as fn
from sklearn.metrics import roc_auc_score,roc_curve

from Network import GraphSAGE
# from DataLoader import parse_data, load_data
from DataLoader import create_ida_graphs # TODO:REMOVE

from os import path
from itertools import chain

import warnings
warnings.filterwarnings("ignore")

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']



class MyModel(object):

    def __init__(self, configs, name=None):
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.configs = configs
        self.name = name

        # self.network = GraphSAGE(configs,3,16).to(self._device)
        self.network = GraphSAGE(configs,1+2*configs.window_size,16).to(self._device)
        self.pred = DotPredictor()

        self.configs.silent = self.configs.wandb or self.configs.silent

        # self.train_data, self.valid_data, self.test_data = create_ida_graphs(valid=True)
        self.train_data, self.test_data = create_ida_graphs()

        # TODO: REMOVE
        # self.input_shape = input_shape
        # configs.input_shape = input_shape

    def train(self, x_train=None, y_train=None, x_valid=None, y_valid=None):

        # Import Data if None
        # if x_train is not None:
        #     if self.configs.validation:
        #         if x_valid is None:
        #             train_data = parse_data(x_train, y_train, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
        #         else:
        #             train_data = parse_data(x_train, y_train, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
        #             train_data = parse_data(x_valid, y_valid, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
        #     else: 
        #         train_data = parse_data(x_train, y_train, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
        # else:
        #     self.train_g, self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g = load_data()
        #     # # TODO: change to dgl
        #     # train_data = torchvision.datasets.CIFAR10(root=self.configs.data_directory, train=True, download=True, transform=transform)
        #     # train_data = torch.utils.data.DataLoader(train_data, batch_size=self.configs.batch_size, shuffle=True, num_workers=2)

        #     # if self.configs.validation:
        #     #     # TODO: change to dgl
        #     #     valid_data = torchvision.datasets.CIFAR10(root=self.configs.data_directory, train=False, download=True, transform=valid_transform)
        #     #     valid_data = torch.utils.data.DataLoader(valid_data, batch_size=self.configs.batch_size, shuffle=False, num_workers=2)

        # train_g, train_pos_g, train_neg_g = load_data(self.configs.data_directory)
        # train_g, train_pos_g, train_neg_g = create_ida_graphs()
        # train_data = (train_pos_g, train_neg_g)

        # Wandb setup
        if self.configs.wandb:
            # Initialize Model Name
            import wandb
            name = self.name if self.name is not None else 'MyModel'
            wandb.init(name=name, config=self.configs, dir=self.configs.log_directory, project="cycle-1", entity="btho-ida")
            wandb.watch(self.network)

        # Number of batches per epoch.  Used for logging to normalize data for epochs
        # num_steps_per_epoch = len(train_data)

        # Optimizer
        if self.configs.adam: optimizer = torch.optim.Adam(chain(self.network.parameters(), self.pred.parameters()), lr=self.configs.learning_rate, weight_decay=self.configs.weight_decay)
        else:                 optimizer = torch.optim.SGD( chain(self.network.parameters(), self.pred.parameters()), lr=self.configs.learning_rate, weight_decay=self.configs.weight_decay, momentum=0.9)

        # Step Scheduler
        if self.configs.step_schedule:
            if self.configs.cosine: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.configs.epochs)
            else:                   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

        # Criterion
        def loss(h,pos_g,neg_g):
            pos_score = self.pred(pos_g, h)
            neg_score = self.pred(neg_g, h)
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
            return F.binary_cross_entropy_with_logits(scores, labels)

        # if self.configs.wandb:
        #     wandb.log({"training accuracy":1/self.configs.num_classes}, step=0)
        #     if self.configs.validation: wandb.log({'validation accuracy':1/self.configs.num_classes, "generalization":1}, step=0)

        if not self.configs.silent: print("--- Training Start ---")
        global_step = 0
        # Epoch Loop
        for i in range(self.configs.epochs):
            total_loss = 0.0
            for x,p,n in self.train_data:
                self.network.train()
                self.pred.train()

                # Compute loss
                h = self.network(x,x.ndata['feat'])
                loss_val = loss(h,p,n)

                # Grad Step
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                total_loss += loss_val.item()

            # Plot loss
            if self.configs.wandb: wandb.log({"epoch":i, "loss": loss_val.item()}, step=i)

            # Step Learning Rate
            if self.configs.step_schedule:
                if self.configs.cosine: scheduler.step()
                else:                   scheduler.step(total_loss)

            # Compute ROC
            ra_score = self.roc_auc(h,p,n)
            if self.configs.validation:
                valid_ra_score = 0
                for vx,vp,vn in self.test_data:
                    vh = self.network(vx,vx.ndata['feat'])
                    valid_ra_score += self.roc_auc(vh,vp,vn)
                valid_ra_score /= len(self.test_data)

            # Print if no logger
            if self.configs.wandb:
                wandb.log({"ROC Score":ra_score}, step=i)
                if self.configs.validation: wandb.log({'validation accuracy':valid_accuracy, "generalization":generalization}, step=global_step)
                if self.configs.step_schedule: wandb.log({'learning rate':optimizer.param_groups[0]['lr']}, step=global_step)
            elif not self.configs.silent: 
                print('Epoch: {:4d} -- Training Loss: {:10.6f} -- ROC Score: {:10.6f}'.format(i, total_loss, ra_score), end='' if self.configs.validation else '\n')
                # if self.configs.validation: print()
                if self.configs.validation: print(' | {:10.6f}'.format(valid_ra_score))

            if self.configs.save_interval is not None and self.configs.save_interval > 0 and i % self.configs.save_interval == 0 and i+1 < self.configs.epochs:
                self.save('i{}'.format(i))

        if self.configs.save_interval is not None and self.configs.save_interval >= 0:
            self.save()
            if not self.configs.silent: print("Saved: True")
        elif not self.configs.silent: print("Saved: False")
        if not self.configs.silent: print("--- Training Complete ---")

    def roc_auc(self,h,pos_g,neg_g,curve=False):
        self.network.eval()
        self.pred.eval()
        pscore = self.pred(pos_g, h).detach()
        nscore = self.pred(neg_g, h).detach()
        labels = torch.cat([torch.ones( pscore.shape[0]),
                            torch.zeros(nscore.shape[0])]).numpy()
        scores = torch.cat([pscore, nscore]).numpy()
        score = roc_auc_score(labels, scores)
        if curve:
            return score, roc_curve(labels, scores)
        return score

    def calc_thresh(self,fp,tp,thresh):
        import pandas as pd
        roc = pd.DataFrame({'tf' : pd.Series(tp-(1-fp), index=np.arange(len(tp))), 'threshold' : pd.Series(thresh, index=np.arange(len(tp)))})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold']) 

    def evaluate(self, x=None, y=None, test_data=None):
        # TODO: quality of life upgrades
        test_data = test_data if test_data else self.test_data 
        self.network.eval()
        self.pred.eval()
        score = 0
        for x,p,n in self.test_data:
            h = self.network(x, x.ndata['feat'])
            rscore,(fp,tp,thresh) = self.roc_auc(h,p,n,curve=True)
            score += rscore
        return score/len(self.test_data), self.calc_thresh(fp,tp,thresh)
        
    def predict(self, x=None):
        if x is None:
            for g,_,_ in self.test_data:
                pass
            return self.network(g,g.ndata['feat'])
        return self.network(x, x.ndata['feat'])

    def save(self, des=''):
        from torch import save
        from os import path
        from pathlib import Path

        # Save the configuration so that at load, different configs will not stop loading
        # TODO: add self.pred
        checkpoint = {'configs':          self.configs,
                      'model_state_dict': self.network.state_dict()}

        # Compute the full file path 
        name = self.name if self.name else ''
        save_dir = path.join(path.dirname(path.abspath(__file__)), self.configs.model_directory, name)
        save_path = path.join(save_dir, 'model'+des+'.th')

        # Ensures the save_directory exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save(checkpoint, save_path)

    def load(self, filePath=''):
        from torch import load
        from os import path

        def update_configs(args):
            for k, v in vars(args).items():
                vars(self.configs)[k] = v

        # Check to see if actual 'th' file passed into load.  If not, it is assumed that the value passed in is a path, and model.th is appended on.
        if filePath[-2:] != 'th': filePath = path.join(filePath,'model.th')
        # Check to see if file passed into load exists.
        if not path.exists(filePath): raise(Exception('Invalid filePath passed into MyModel.load: "{}" [{}]'.format(str(filePath), str(type(filePath)))))

        checkpoint = load(path.join(path.dirname(path.abspath(__file__)), filePath), map_location='cpu')
        update_configs(checkpoint['configs'])
        # TODO: add self.pred
        self.network = MyNetwork(self.configs).to(self._device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        return self.configs


