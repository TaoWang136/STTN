import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.transformer_encoder import T_STGCN

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))
        
        
    def forward(self, x):
        if self.c_in > self.c_out:

            x = self.align_conv(x)
        elif self.c_in < self.c_out:

            batch_size, _, timestep, n_vertex = x.shape
            
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:

            x = x
        return x
    
class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out

        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        #print('c_in, c_out',c_in, c_out)
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()

        self.model = T_STGCN(1,self.c_in,128)#N,k,c_model_d
        
    def forward(self, x):
        

        x_res =self.align(x)#[32, 16, 12, 67]>[32, 64, 12, 67]

        x_e = x.permute((0,2,1,3)).float()

        pred = self.model(x_e).permute((0,3,2,1))#[32, 12, 16, 67]>[32, 128, 12, 67]

        x_p = pred[:, : 64, :, :]

        x_q = pred[:, -64:, :, :]

        x = torch.mul((x_p + x_res), self.sigmoid(x_q))#>[32, 64, 12, 67],x_p;x_q;x_res这三者都是[32, 64, 12, 67]
        #x=torch.add(pred,x_res)

        return x
    
    
    
class TemporalConvLayer_outblock(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer_outblock, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_in_1=self.align(x)


        x_in = x_in_1[:, :, self.Kt - 1:, :] 

        x_causal_conv = self.causal_conv(x)#[48, 128, 1, 228]>[48, 256, 1, 228]

        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        print('x_p,x_q',x_p.shape,x_q.shape)
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))
        return x
    
    
class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))##([3, 16, 16])
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))##16
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    def forward(self, x):#([32, 16, 10, 228])Ks=3
        x = x.permute(0, 2, 3, 1)

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x#[32, 10, 228, 16],gso=([228, 228])
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            #print('x_0x_0x_0x_0',x_0.shape,x_1.shape)
            for k in range(2, self.Ks):# 2
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])##切比雪夫多项式
        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)##32, 16, 10, 228
        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv
        return cheb_graph_conv
    
class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
    def forward(self, x):

        x_gc_in = self.align(x)

        x_gc = self.cheb_graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        
        return x_gc_out
    
class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()

        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)##将1变成64

        self.graph_conv = GraphConvLayer(graph_conv_type, 1, channels[1], Ks, gso, bias)#####如果用三明治结构需要将，channels[0]改成1，，将1变成16

        self.tmp_conv2 = TemporalConvLayer(Kt,channels[1], channels[2], n_vertex, act_func)#将16变成64
        
        self.graph_conv2 = GraphConvLayer(graph_conv_type,64,64, Ks, gso, bias)#####如果用三明治结构需要将，channels[0]改成1
        

        
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        
    def forward(self, x):
        ###在这加
        
        #print('x1',x.shape)
        #x = self.tmp_conv1(x)
        print('x2',x.shape)#[36, 1, 12, 228]
        x = self.graph_conv(x)
        print('x3',x.shape)#([36, 16, 12, 228]
        #x = self.relu(x)
        ##在这加

        x = self.tmp_conv2(x)#>[32, 16, 12, 67]

        # x = self.graph_conv2(x)

        print('x4',x.shape)#[36, 64, 12, 228]
        
        
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = self.dropout(x)
        #[36, 64, 12, 228]
        return x

class My_OutputBlock(nn.Module):
    def __init__(self, endout,adj,his,drop):
        super(My_OutputBlock, self).__init__()
        self.endout=endout
        self.tmp_conv1 = TemporalConvLayer_outblock(his, 64, 128, adj, 'glu')
        self.tc1_ln = nn.LayerNorm([adj,128])
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(64,1)

        
    def forward(self, x):

        x = self.tmp_conv1(x)

        #print('xxxxxxxxxxxxxxx',x.shape)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        #x = self.relu(x)
        #print('fff',x.shape)
        x = x.permute(0, 3, 1, 2)
        #print('x5',x.shape)
        return x