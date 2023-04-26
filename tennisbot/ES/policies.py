"""
Source repo: https://github.com/enajx/ES
"""

import torch
import torch.nn as nn

       
class MLP(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space_dim, action_space_dim, bias=False):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_space_dim, 128, bias=bias) 
        self.linear2 = nn.Linear(128, 64, bias=bias)
        self.out = nn.Linear(64, action_space_dim, bias=bias)

    def forward(self, ob):
        state = torch.as_tensor(ob).float().detach()    
        x = torch.tanh(self.linear1(state))   
        x = torch.tanh(self.linear2(x))
        o = self.out(x)
        return o.squeeze()

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters()).detach().numpy()
    
    
class CNN(nn.Module):
    "CNN+MLP"
    def __init__(self, input_channels, action_space_dim):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=False)   
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)
        
        self.linear1 = nn.Linear(648, 128, bias=False) 
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.out = nn.Linear(64, action_space_dim, bias=False)

    def forward(self, ob):
        
        state = torch.as_tensor(ob.copy())
        state = state.float()
        
        x = self.pool(torch.tanh(self.conv1(state)))
        x = self.pool(torch.tanh(self.conv2(x)))
        
        x = x.view(-1)
        
        x = torch.tanh(self.linear1(x))   
        x = torch.tanh(self.linear2(x))
        o = self.out(x)
        return o

    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters() ).detach().numpy()

import torch
import torch.nn as nn
import torch.nn.functional as F


DOF = 6


class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 input_c , action_dim
            ):
        self.input_c = input_c
        self.action_dim = action_dim

        super(GatedCNN, self).__init__()
        # self.res_block_count = res_block_count
        # self.embd_size = embd_size

        #self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...



        self.conv_0 = nn.Conv1d(in_channels=  self.input_c, out_channels= 8, kernel_size=2, dilation= 1,stride=1,padding='valid')
        # self.b_0 = nn.Parameter(torch.randn(size= (8,16)))
        # self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.conv_gate_0 = nn.Conv1d(in_channels=  self.input_c, out_channels=8, kernel_size=2, dilation=1, stride=1)
        # self.c_0 = nn.Parameter(torch.randn(size= (8,16)))


        self.conv_1 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=2, dilation=2, stride=1)
        # self.b_1 = nn.Parameter(torch.randn(size=(12, 32)))
        self.conv_gate_1 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=2, dilation=2, stride=1)
        # self.c_1 = nn.Parameter(torch.randn(size=(12, 32)))


        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=self.action_dim , kernel_size=2, dilation=4, stride=1)



    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        # bs = x.size(0) # batch size
        # seq_len = x.size(1)
        # x = self.embedding(x) # (bs, seq_len, embd_size)
        #
        # # CNN
        # x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)


        x = torch.from_numpy(x).float()
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        # A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        # B += self.c_0.repeat(1, 1, seq_len, 1)
        h = F.tanh(A) * F.sigmoid(B)  # (bs, Cout, seq_len, 1)



        A = self.conv_1(h)  # (bs, Cout, seq_len, 1)
        # A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_1(h)  # (bs, Cout, seq_len, 1)
        # B += self.c_0.repeat(1, 1, seq_len, 1)
        h = F.tanh(A) * F.sigmoid(B)  # (bs, Cout, seq_len, 1)

        h = self.conv_2(h)


        return h.squeeze()
    def get_weights(self):
        return  nn.utils.parameters_to_vector(self.parameters() ).detach().numpy()
