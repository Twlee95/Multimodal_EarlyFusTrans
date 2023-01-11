import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
# input_window = 100 # number of input steps
# output_window = 1 # number of prediction steps, in this model its fixed to one
# batch_size = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=10, max_len=5000):
        super(PositionalEncoding, self).__init__()
        if (d_model % 2) == 0:
            pe = torch.zeros(max_len, d_model) # 5000, 10
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 5000,1 
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [5]
            # position * div_term : 5000,5
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)  # torch.Size([max_len, 1, d_model]) # torch.Size([5000, 1, 2])
        else:
            pe = torch.zeros(max_len, d_model) # 5000, 5
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 5000,1 
            sin_div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [5]
            cos_div_term = torch.exp(torch.arange(0, d_model-1, 2).float() * (-math.log(10000.0) / d_model)) # [5]
            # position * div_term : 5000,3
            pe[:, 0::2] = torch.sin(position * sin_div_term)
            pe[:, 1::2] = torch.cos(position * cos_div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
        
        ''' 
        RuntimeError: The expanded size of the tensor (2) 
        must match the existing size (3) at non-singleton dimension 1.  
        Target sizes: [5000, 2].  Tensor sizes: [5000, 3]
        '''
        # torch.Size([max_len, 1, d_model]) # torch.Size([5000, 1, 2])
        # pe.requires_grad = False
        self.register_buffer('pe', pe) ## 매개변수로 간주하지 않기 위한 것

    def forward(self, x):
        # pe :torch.Size([5000, 1, 2])
        # x : torch.Size([10, 64, 2])
        # self.pe[:x.size(0), :] : torch.Size([10, 1, 2])

        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self,input_feature_size,
                Transformer_feature_size, nhead,args, 
                num_layers=1, dropout=0.1, x_frames = 10):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.args =args
        
        ## ----- Modality1 Transformer ------ ##
        self.input_linear = nn.Linear(input_feature_size,Transformer_feature_size)
        self.pos_encoder = PositionalEncoding(Transformer_feature_size,5000)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=Transformer_feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(x_frames,1)


        self.init_weights()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sigmoid = nn.Sigmoid()

        
        self.Multimodal_linear = nn.Linear(Transformer_feature_size,1)


    def init_weights(self):
        initrange = 0.1

        self.input_linear.bias.data.zero_()
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src):
        
        #print(src1.size()) # [64, 10, 17]
        #print(src2.size()) # [64, 10, 17]
        #print(src3.size()) # [64, 10, 6]
        
        src = self.input_linear(src)
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src) # [64, 10, 40]
        encoder_output = self.transformer_encoder(src) #, self.src_mask) [64, 10,64]
        transformer_output = self.decoder(encoder_output.transpose(1,2)).squeeze()  # 64, 64
        output = self.Multimodal_linear(transformer_output).squeeze()
        output = self.sigmoid(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask