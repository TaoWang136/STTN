import torch
import torch.nn as nn

from model import layers

class STGCNChebGraphConv(nn.Module):

    def __init__(self, args, blocks, n_vertex):#block[[1], [64, 16, 64], [64, 16, 64], [128, 128], [1]]
        super(STGCNChebGraphConv, self).__init__()
        ###两个STConv Block，每个block中有一个三明治
        modules = []
        for l in range(len(blocks) - 3):
            print('len(blocks)len(blocks)len(blocks)',len(blocks))
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
            print('d',blocks[l][-1])
    
        self.st_blocks = nn.Sequential(*modules)
        
        ###outputlayer
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)#4

        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.My_OutputBlock(1,n_vertex,args.n_his,args.droprate)


    def forward(self, x):
        print('x',x.shape)
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)

        return x