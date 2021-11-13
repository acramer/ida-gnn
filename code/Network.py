### YOUR CODE HERE
import torch
import torch.nn as nn
import torch.nn.functional as F

"""This script defines the network.
"""
def MyNetwork(configs):
    nets = {'basic':            BasicNet,
            'res_skip':         ResSkipNet,
            'efficient':        EfficientNet,
            'parallel_channel': ParallelChannelNet,
            'helix':            HelixNet,
            'flat_coding':      FlatCodingNet,
            'multi_loop':       MultiLoopNet,
            'dla':              DLA,
           }
    return nets[configs.architecture](configs)


# Basic Network used mainly for testing setup
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


# ResSkipNet
class ResSkipNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(configs.num_channels, 32,7,1,7//2),
            torch.nn.ReLU(),
        )
        self.r1 = RBlock(    32,    64,1)
        self.r2 = RBlock(    64,    64,1)
        self.r3 = RBlock( 64+64,   128,1)
        self.r4 = RBlock(128+32,128+32,1)
        self.foot = torch.nn.Sequential(
            torch.nn.Conv2d(128+32+configs.num_channels,128+32+configs.num_channels,5,2,5//2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128+32+configs.num_channels,128+32+configs.num_channels,5,2,5//2),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(8),
            torch.nn.Conv2d(128+32+configs.num_channels, configs.num_classes, 1)
        )
    
    def forward(self, inputs):
        out1 = self.head(inputs)
        out2 = self.r1(out1)
        out3 = torch.cat((self.r2(out2), out2),   dim=1)
        out4 = torch.cat((self.r3(out3), out1),   dim=1)
        out5 = torch.cat((self.r4(out4), inputs), dim=1)
        out  = self.foot(out5)
        return out.reshape(-1,self.configs.num_classes)



# EfficientNet Implementation
class EfficientNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        if self.configs.activation == 'swish':
            activation = Swish
        else:
            activation = torch.nn.ReLU

        residual=True
    
        # Efficient Net Default Stack Configs
        in_channels =  [32,16,24,40, 80,112,192]
        out_channels = [16,24,40,80,112,192,320]
        kernel =       [ 3, 3, 5, 3,  5,  5,  3]
        stride =       [ 1, 2, 2, 2,  1,  2,  1]
        blocks =       [ 1, 2, 2, 3,  3,  4,  1]
        types  =       [ 1, 6, 6, 6,  6,  6,  6]
    
        # Dropout networks increase in dropout rate uniformly from 0 to 0.2 for each block
        dropouts = [self.configs.dropout_rate*i/sum(blocks) for i in range(sum(blocks)+1)]
    
        # First layers before MBConv Stacks are added
        layers = [torch.nn.Conv2d(self.configs.num_channels, 32, 3, 2, 3//2),
                  torch.nn.BatchNorm2d(32),
                  activation(),
                 ]
    
        # MBConv Stacks added
        for cin, cout, ksize, stride, num_block, block_type in zip(in_channels, out_channels, kernel, stride, blocks, types):
            Block = MBConv6
            if   block_type == 1: Block = MBConv1
    
            droprate = dropouts.pop(0)
            if stride > 1: droprate = 0
            layers.append(Block(cin, cout, ksize, stride, activation, residual, droprate))
    
            for _ in range(1, num_block): layers.append(Block(cout, cout, ksize, 1, activation, residual, dropouts.pop(0)))
    
        # Final layers
        layers.extend([
                  torch.nn.Conv2d(320,1280,1,1,1//2),
                  torch.nn.BatchNorm2d(1280),
                  torch.nn.AdaptiveAvgPool2d(1),
                  Flatten(),
                  torch.nn.Dropout(dropouts.pop(0)),
                  torch.nn.Linear(1280,self.configs.num_classes),
                 ])
    
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)



# Parallel Network Implementation
class ParallelChannelNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        channels=[32, 24, 64, 128]
        layers = [torch.nn.Conv2d(self.configs.num_channels, 32, 5, 1, 5//2),
                  torch.nn.BatchNorm2d(channels[0]),
                  Swish(),
                 ]
    
        cl = channels[0]
        stride_count = 0
        for ci in channels[1:]:
            stride = 2 if stride_count < 2 else 1
            layers.append(PBlock(cl, ci, 3, stride))
            layers.append(PBlock(ci, ci, 3, 1))
            cl = ci
            stride_count += 1
    
        # Final layers
        layers.extend([
                  torch.nn.Conv2d(channels[-1],512,1,1,1//2),
                  torch.nn.BatchNorm2d(512),
                  torch.nn.AdaptiveAvgPool2d(1),
                  Flatten(),
                  torch.nn.Linear(512,self.configs.num_classes),
                 ])
    
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)



# Helix Network Implementation
class HelixNet(torch.nn.Module):
    def __init__(self, configs, ):
        super().__init__()
        self.configs = configs
        channels=[32, 24, 64, 128]
        layers = [torch.nn.Conv2d(self.configs.num_channels, channels[0], 5, 1, 5//2),
                  torch.nn.BatchNorm2d(channels[0]),
                  Swish(),
                 ]
    
        cl = channels[0]
        stride_count = 0
        for ci in channels[1:-1]:
            stride = 2 if stride_count < 2 else 1
            layers.append(HBlock(cl, ci, 3, 3, 3, stride))
            layers.append(HBlock(ci, ci, 3, 3, 3, 1))
            stride_count += 1
            cl = ci
        ci = channels[-1]
        layers.append(HBlock(cl, ci, 3, 3, 3, 1))
        layers.append(HBlock(ci, ci, 3, 3, 3, 1, True))
    
        # Final layers
        layers.extend([
                  torch.nn.Conv2d(channels[-1],512,1,1,1//2),
                  torch.nn.BatchNorm2d(512),
                  torch.nn.AdaptiveAvgPool2d(1),
                  Flatten(),
                  torch.nn.Linear(512,self.configs.num_classes),
                 ])
    
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)



# FlatEncoding Network Implementation
class FlatCodingNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(self.configs.num_channels, 32, 5, 1, 5//2),
            torch.nn.BatchNorm2d(32),
            Swish(),
        )
            
        self.f1 = FBlock( 32, 24, 3, 2)
        self.f2 = FBlock( 24, 24, 3, 1)
        self.f3 = FBlock( 24, 64, 3, 2)
        self.f4 = FBlock( 64, 64, 3, 1)
        self.f5 = FBlock( 64,128, 3, 1)
        self.f6 = FBlock(128,128, 3, 1)
    
        # Final layers
        self.foot = torch.nn.Sequential(
            torch.nn.Conv2d(blocks[-1],512,1,1,1//2),
            torch.nn.BatchNorm2d(512),
            torch.nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
    
        self.classifier = torch.nn.Linear(512+6*128,self.configs.num_classes)
    
    def forward(self, inputs):
        o0     = self.head(inputs)
        o1, c1 = self.f1(o0)
        o2, c2 = self.f2(o1)
        o3, c3 = self.f3(o2)
        o4, c4 = self.f4(o3)
        o5, c5 = self.f5(o4)
        o6, c6 = self.f6(o5)
        out = self.foot(o5)
        return self.classifier(torch.cat((c1,c2,c3,c4,c5,c6,out),1))


# MultiLoop Network Implementation
class MultiLoopNet(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        channels=[32, 256, 512]

        layers = [torch.nn.Conv2d(self.configs.num_channels, channels[0], 5, 1, 5//2),
                  torch.nn.BatchNorm2d(channels[0]),
                  Swish(),
                 ]
    
        cl = channels[0]
        stride_count = 0
        for ci in channels[1:]:
            stride = 2 if stride_count < 2 else 1
            layers.append(MultiLoopBlock(cl, ci, 3, stride))
            stride_count += 1
            cl = ci
    
        # Final layers
        layers.extend([
                  torch.nn.Conv2d(blocks[-1],512,1,1,1//2),
                  torch.nn.BatchNorm2d(512),
                  torch.nn.AdaptiveAvgPool2d(1),
                  Flatten(),
                  torch.nn.Linear(512,self.configs.num_classes),
                 ])
    
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)



# DLA implementation
class DLA(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        if self.configs.activation == 'swish':
            activation = Swish
        else:
            activation = torch.nn.ReLU
    
        # Uniform increments from 0 to dropout_rate for all TBlocks
        dropouts = [self.configs.dropout_rate*i/4 for i in range(1,5)]

        Block = getBlock(self.configs.block_type)
    
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.configs.num_channels, 16, 3, 1, 3//2, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 16, 3, 1, 3//2, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, 3, 1, 3//2, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            TBlock( 32,  64,                      1, 1, False, activation, dropouts.pop(0), NodeBlock=Block, SE=self.configs.squeeze_excite),
            TBlock( 64, 128,                      2, 2, True,  activation, dropouts.pop(0), NodeBlock=Block, SE=self.configs.squeeze_excite),
            TBlock(128, 256, self.configs.dla_depth, 2, True,  activation, dropouts.pop(0), NodeBlock=Block, SE=self.configs.squeeze_excite),
            TBlock(256, 512,                      1, 2, True,  activation, dropouts.pop(0), NodeBlock=Block, SE=self.configs.squeeze_excite),
            torch.nn.AvgPool2d(4),
            Flatten(),
            torch.nn.Linear(512, self.configs.num_classes))
    
    def forward(self, inputs):
            return self.net(inputs)



# -----------------------------------------
# Blocks
# -----------------------------------------

def getBlock(name):
    blocks = { 'mbconv1': MBConv1,
               'mbconv6': MBConv6,
               'pblock' : PBlock,
               'dblock' : DBlock,
             }
    return blocks[name]

# BasicNet Block
class BasicBlock(torch.nn.Module):
    def __init__(self, cin, cout, stride=2, ksize=3):
        super().__init__()
        self.net = torch.nn.Sequential( torch.nn.Conv2d(cin,  cout, ksize, stride, ksize//2),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(cout, cout, 3,     1, 3//2),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(cout, cout, 3,     1, 3//2),
                                        torch.nn.ReLU(),
                                        )
    def forward(self, inputs):
        return self.net(inputs)

# ResSkip Block
class RBlock(torch.nn.Module):
    def __init__(self, cin, cout, stride=2, ksize=3):
        super().__init__()
        self.resdown = False
        if stride>1 or cin != cout: self.resdown = torch.nn.Conv2d(cin, cout, 1, stride)
        self.net = torch.nn.Sequential( torch.nn.Conv2d(cin,  cout, ksize, stride, ksize//2),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(cout, cout, 3,     1, 3//2),
                                        torch.nn.ReLU(),
                                        )
    def forward(self, inputs):
        if self.resdown:
            return self.net(inputs) + self.resdown(inputs)
        return self.net(inputs) + inputs

# EfficientNet Blocks
class MBConv1(torch.nn.Module):
    def __init__(self, cin, cout, ksize=3, stride=1, activation=torch.nn.ReLU, residual=True, dropout_rate=0):
        super().__init__()

        self.residual = residual
        self.projection = None
        if residual and (cin != cout or stride > 1): self.projection = torch.nn.Conv2d(cin,cout,1,stride,1//2)

        self.net = torch.nn.Sequential( torch.nn.Conv2d(cin, cin, ksize, stride, ksize//2, groups=cin),
                                        torch.nn.BatchNorm2d(cin),
                                        activation(),
                                        SqueezeExcitation(cin, cin//4),
                                        torch.nn.Conv2d( cin, cout,     1,      1,     1//2),
                                        torch.nn.BatchNorm2d(cout),
                                        #torch.nn.Dropout(dropout_rate),
                                        DropConnect(dropout_rate),
                                        )

    def forward(self, inputs):
        if self.residual:
            res = inputs
            if self.projection: res = self.projection(inputs)
            return self.net(inputs) + res
        return self.net(inputs)

class MBConv6(torch.nn.Module):
    def __init__(self, cin, cout, ksize=3, stride=1, activation=torch.nn.ReLU, residual=True, dropout_rate=0):
        super().__init__()

        self.residual = residual
        self.projection = None
        if residual and (cin != cout or stride > 1): self.projection = torch.nn.Conv2d(cin,cout,1,stride,1//2)

        cmid = 6*cin
        self.net = torch.nn.Sequential( torch.nn.Conv2d( cin, cmid,     1,      1,     1//2),
                                        torch.nn.BatchNorm2d(cmid),
                                        activation(),
                                        torch.nn.Conv2d(cmid, cmid, ksize, stride, ksize//2, groups=cmid),
                                        torch.nn.BatchNorm2d(cmid),
                                        activation(),
                                        SqueezeExcitation(cmid, cin//4),
                                        torch.nn.Conv2d(cmid, cout,     1,      1,     1//2),
                                        torch.nn.BatchNorm2d(cout),
                                        #torch.nn.Dropout(dropout_rate),
                                        DropConnect(dropout_rate),
                                        )

    def forward(self, inputs):
        if self.residual:
            res = inputs
            if self.projection: res = self.projection(inputs)
            return self.net(inputs) + res
        return self.net(inputs)


# ParallelNet Blocks
class PBlock(torch.nn.Module):
    def __init__(self, cin, cout, ksize=3, stride=1, activation=torch.nn.ReLU, dropout_rate=0):
        super().__init__()

        assert cout % 2 == 0

        up_layers = [
            torch.nn.Conv2d(  cin*1,   cin*8,     1,      1,     1//2),
            torch.nn.BatchNorm2d(cin*8),
            activation(),
            torch.nn.Conv2d(  cin*8,   cin*8, ksize, stride, ksize//2),
            torch.nn.BatchNorm2d(cin*8),
            activation(),
            torch.nn.Conv2d(  cin*8, cout//2,     1,      1,     1//2),
        ]
        down_layers = [
            torch.nn.Conv2d(  cin*1,  cin//8,     1,      1,     1//2),
            torch.nn.BatchNorm2d(cin//8),
            activation(),
            torch.nn.Conv2d( cin//8,  cin//8, ksize, stride, ksize//2),
            torch.nn.BatchNorm2d(cin//8),
            activation(),
            torch.nn.Conv2d( cin//8, cout//2,     1,      1,     1//2),
        ]
        layers = [
            activation(),
            torch.nn.BatchNorm2d(cout),
            torch.nn.Dropout(dropout_rate),
        ]
        self.up_net   = torch.nn.Sequential(*up_layers)
        self.down_net = torch.nn.Sequential(*down_layers)
        self.net      = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(torch.cat((self.up_net(inputs),self.down_net(inputs)), 1))
        #out1 = self.up_net(inputs)
        #out2 = self.down_net(inputs)
        #print('out1',out1.shape)
        #print('out2',out2.shape)
        #out3 = torch.cat((out1,out2), 1)
        #print('out3',out3.shape)
        #out4 = self.net(out3)
        #print('out4',out4.shape)
        #return out4
    

# HelixNet Blocks
class HBlock(torch.nn.Module):
    def __init__(self, cin, cout, cx, ksize, xksize, stride, merge=False):
        super().__init__()
        self.merge = merge

        if merge:
            assert cout % 4 == 0
            cout = cout//2
        assert cout % 2 == 0

        up_layers = [
            torch.nn.Conv2d(  cin*1,   cin*2, ksize, stride, ksize//2),
            Swish(),
            torch.nn.Conv2d(  cin*2,   cin*4, ksize, 1, ksize//2),
            Swish(),
            torch.nn.Conv2d(  cin*4,   cin*8, ksize, 1, ksize//2),
            Swish(),
            torch.nn.Conv2d(  cin*8,    cout, ksize, 1, ksize//2),
        ]
        down_layers = [
            torch.nn.Conv2d(  cin*1,  cin//2, ksize, stride, ksize//2),
            Swish(),
            torch.nn.Conv2d( cin//2,  cin//4, ksize, 1, ksize//2),
            Swish(),
            torch.nn.Conv2d( cin//4,  cin//8, ksize, 1, ksize//2),
            Swish(),
            torch.nn.Conv2d( cin//8,    cout, ksize, 1, ksize//2),
        ]
        self.up_net   = torch.nn.Sequential(*up_layers)
        self.down_net = torch.nn.Sequential(*down_layers)

        if merge:
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(cout*2, cout*2, ksize, 1, ksize//2),
                Swish(),
                torch.nn.BatchNorm2d(cout*2)
            )
        else:
            self.up_to_up     = torch.nn.Conv2d(cout, cout-cx, xksize, 1, xksize//2)
            self.down_to_up   = torch.nn.Conv2d(cout,      cx, xksize, 1, xksize//2)
            self.down_to_down = torch.nn.Conv2d(cout, cout-cx, xksize, 1, xksize//2)
            self.up_to_down   = torch.nn.Conv2d(cout,      cx, xksize, 1, xksize//2)

            self.foot_up   = torch.nn.Sequential(Swish(), torch.nn.BatchNorm2d(cout))
            self.foot_down = torch.nn.Sequential(Swish(), torch.nn.BatchNorm2d(cout))

    def forward(self, inputs):
        if not isinstance(inputs, tuple): inputs = (inputs, inputs) 
        uin, din = inputs
        if self.merge:
            return self.net(torch.cat((self.up_net(uin),self.down_net(din)),1))
        else:
            uout, dout = self.up_net(uin), self.down_net(din)
            up   = torch.cat((    self.up_to_up(uout), self.down_to_up(dout)), 1)
            down = torch.cat((self.down_to_down(dout), self.up_to_down(uout)), 1)
            return self.foot_up(up), self.foot_down(down)


# FlatEncoding Block
class FBlock(torch.nn.Module):
    def __init__(self, cin, cout, ksize, stride, flatout=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d( cin, cout, ksize, stride, ksize//2),
            Swish(),
            torch.nn.Conv2d(cout, cout,     3,      1, ksize//2),
            Swish(),
            torch.nn.Conv2d(cout, cout,     3,      1, ksize//2),
            Swish(),
        )
        self.flat = torch.nn.Sequential(
            torch.nn.BatchNorm2d(cout),
            torch.nn.AdaptiveAvgPool2d(1),
            Flatten(),
            torch.nn.Linear(cout, flatout),
        )

    def forward(self,inputs):
        outs = self.net(inputs)
        return outs, self.flat(outs)


# MultiLoopNet Block
class MultiLoopBlock(torch.nn.Module):
    def __init__(self, cin, cout, ksize, stride, flatout=128):
        super().__init__()
        assert cout % 4 == 0
        self.head = torch.nn.Sequential(torch.nn.Conv2d( cin, cout//4, ksize, stride, ksize//2), Swish())
        self.net  = torch.nn.Sequential(
            torch.nn.Conv2d(cout//4, cout//2, ksize, 1, ksize//2),
            Swish(),
            torch.nn.Conv2d(cout//2,    cout,     3, 1, ksize//2),
            Swish(),
            torch.nn.Conv2d(   cout, cout//4,     3, 1, ksize//2),
            Swish(),
        )
        self.foot = torch.nn.Sequential(Swish(), torch.nn.BatchNorm2d(cout))

    def forward(self,inputs):
        z = self.head(inputs)
        outs = []
        for _ in range(4):
            z = self.net(z)
            outs.append(z)
        return self.foot(torch.cat(outs,1))

# DLA Tree Block
class DBlock(torch.nn.Module):
    def __init__(self, cin, cout, ksize=3, stride=1, activation=torch.nn.ReLU, dropout_rate=0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(cin,  cout, ksize, stride, ksize//2, bias=False),
            torch.nn.BatchNorm2d(cout),
            activation(),
            torch.nn.Conv2d(cout, cout, ksize,      1, ksize//2, bias=False),
            torch.nn.BatchNorm2d(cout),
            torch.nn.Dropout(dropout_rate),
            )
        if stride == 1 and cin == cout: self.projection = None
        else: self.projection = torch.nn.Sequential(torch.nn.Conv2d(cin, cout, 1, stride, 1//2, bias=False),torch.nn.BatchNorm2d(cout))
        self.f_act = activation()

    def forward(self, inputs):
        res = inputs
        if self.projection: res = self.projection(inputs)
        return self.f_act(self.net(inputs) + res)

# DLA Tree Block
class TBlock(torch.nn.Module):

    class ANode(torch.nn.Module):
        def __init__(self, cin, cout, ksize=1, activation=torch.nn.ReLU, SE=False):
            super().__init__()
            layers = [torch.nn.Conv2d(cin, cout, ksize, 1, ksize//2, bias=False),
                      torch.nn.BatchNorm2d(cout), activation()]
            if SE: layers.insert(1,SqueezeExcitation(cout,cout//4))
            self.net = torch.nn.Sequential(*layers)
        def forward(self, inputs):
            return self.net(torch.cat(inputs,1))

    def __init__(self, cin, cout, depth, stride=1, ida=False, activation=torch.nn.ReLU, dropout_rate=0, NodeBlock=DBlock, SE=False):
        super().__init__()

        TBlock = self.__class__ 
        clast = cin
        self.ida = ida

        num_nodes = 2 + (depth-1) + int(self.ida)
        num_dropout = 2 + (depth-1) # All nodes except the ida get a dropout
        dropouts = [dropout_rate*i/num_dropout for i in range(1,num_dropout+1)] 


        self.anode = self.ANode(num_nodes*cout, cout, activation=activation, SE=SE)
        if self.ida: self.ida = NodeBlock(clast,cout,stride=stride,activation=activation)
        self.nodes = []
        for i in range(depth-1,0,-1):
            tnode = TBlock(clast,cout,i,stride,activation=activation, dropout_rate=dropouts.pop(0), NodeBlock=NodeBlock)
            self.__setattr__('tree_{}'.format(i), tnode)
            self.nodes.append(tnode)
            clast = cout
            stride = 1

        self.node_1 = NodeBlock(clast,cout,stride=stride,activation=activation, dropout_rate=dropouts.pop(0))
        self.node_2 = NodeBlock(cout, cout,stride=1,     activation=activation, dropout_rate=dropouts.pop(0))
        self.nodes.append(self.node_1)
        self.nodes.append(self.node_2)

    def forward(self, inputs):
        outputs = [inputs]
        for node in self.nodes:
            outputs.append(node(outputs[-1]))
        outputs.pop(0)
        if self.ida: outputs.insert(0,self.ida(inputs))
        return self.anode(outputs)






# Helper Networks
class Flatten(torch.nn.Module):
    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)

class Swish(torch.nn.Module):
    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)

class SqueezeExcitation(torch.nn.Module):
    def __init__(self, cin, cout, activation=Swish):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(cin,  cout, 1, 1, 1//2),
            activation(),
            torch.nn.Conv2d(cout,  cin, 1, 1, 1//2),
            torch.nn.Sigmoid(),
        )
    def forward(self, inputs):
        outs = torch.mean(inputs, dim=(-2, -1), keepdim=True)
        outs = self.net(outs)
        return outs * inputs

class DropConnect(torch.nn.Module):
    def __init__(self, drop_rate=0.2):
        super().__init__()
        self.keep_rate = 1 - drop_rate
        self.forward = self.shortcut
        if self.keep_rate != 1: self.forward = self.dropforward

    def shortcut(self, inputs):
        return inputs

    def dropforward(self, inputs):
        drop_mask = torch.floor(torch.rand(inputs.shape[0],1,1,1) + self.keep_rate).type_as(inputs) / self.keep_rate
        return drop_mask * inputs

### END CODE HERE
