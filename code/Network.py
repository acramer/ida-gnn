import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv

class BasicNet(torch.nn.Module):
    def __init__(self, configs, blocks=[64,128]):
        super().__init__()
        self.configs = configs
        layers = [torch.nn.Conv2d(configs.num_channels, 2*configs.num_channels,7,2,7//2), torch.nn.ReLU()]
        lc = 2*configs.num_channels
        for c in blocks:
            layers.append(BasicBlock(lc,c))
            lc = c
        layers.append(torch.nn.Conv2d(lc, configs.num_classes, 32//(2**(1+len(blocks))) ))
    
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs).reshape(-1,self.configs.num_classes)

class GraphSAGE(nn.Module):
    def __init__(self, configs, in_feats, h_feats):
        super().__init__()
        self.configs = configs
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

