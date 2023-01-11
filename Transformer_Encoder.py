import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
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
            pe = torch.zeros(max_len, 40) # 5000, 10
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 5000,1 
            div_term = torch.exp(torch.arange(0, 40, 2).float() * (-math.log(10000.0) / 40)) # [5]
            # position * div_term : 5000,5
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)  # torch.Size([max_len, 1, d_model]) # torch.Size([5000, 1, 2])
        else:
            pe = torch.zeros(max_len, 40) # 5000, 5
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 5000,1 
            sin_div_term = torch.exp(torch.arange(0, 40, 2).float() * (-math.log(10000.0) / 40)) # [5]
            cos_div_term = torch.exp(torch.arange(0, 40-1, 2).float() * (-math.log(10000.0) / 40)) # [5]
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
                Transformer_feature_size, nhead,args, dropout=0.1, x_frames = 10):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.args =args
        
        ## ----- Modality1 Transformer ------ ##
        self.pos_encoder = PositionalEncoding(10,5000)
        self.encoder_layer = EncoderLayer()
        self.decoder = nn.Linear(x_frames,1)


        self.init_weights()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sigmoid = nn.Sigmoid()

        
        self.Multimodal_linear = nn.Linear(40,1)


    def init_weights(self):
        initrange = 0.1

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src):
        
        #print(src1.size()) # [64, 10, 17]
        #print(src2.size()) # [64, 10, 17]
        #print(src3.size()) # [64, 10, 6]

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        
        src = self.pos_encoder(src) # torch.Size([32, 10, 40]
        
        #print(src.size()) #torch.Size[32, 10, 40]
        encoder_output, attn_prob= self.encoder_layer(src) #, self.src_mask) 
        #print(attn_prob.size()) torch.Size([32, 4, 10, 10])
        #print(encoder_output.size()) # torch.Size([32, 10, 64])
        transformer_output = self.decoder(encoder_output.transpose(1,2)).squeeze()  # 64, 64

        output = self.Multimodal_linear(transformer_output).squeeze()
        
        output = self.sigmoid(output)

        return output,attn_prob

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

# EncoderLayer(d_model=Transformer_feature_size, nhead=nhead, dropout=dropout)

""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attn = MultiHeadAttention(40,6,40)
        self.layer_norm1 = nn.LayerNorm(40, eps=1e-12)
        self.pos_ffn = PoswiseFeedForwardNet()
        self.layer_norm2 = nn.LayerNorm(40, eps=1e-12)
    
    def forward(self, inputs):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)

        # [32, 64, 10]
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob






""" multi head attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidn, n_head, d_head):
        super().__init__()
        self.d_hidn = d_hidn
        self.n_head = n_head
        self.d_head = d_head

        self.W_Q = nn.Linear(d_hidn, n_head * d_head)
        self.W_K = nn.Linear(d_hidn, n_head * d_head)
        self.W_V = nn.Linear(d_hidn, n_head * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(d_head)
        self.linear = nn.Linear(n_head * d_head, d_hidn)
    
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)

        # print(Q.size()) #torch.Size[32, 10, 40]) torch.Size([32, 6, 10, 40])


        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2).transpose(-1, -2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2).transpose(-1, -2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2).transpose(-1, -2)
        
        #print(q_s.size()) # ([32, 6, 40, 10]
        #print(k_s.size()) #([32, 6, 40, 10]
        #print(v_s.size()) # ([32, 6, 40, 10]
        # (bs, n_head, n_q_seq, n_k_seq)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)
        
        # print(attn_prob.size()) # [32, 6, 40, 40]
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 3).contiguous().view(batch_size, -1, self.n_head * self.d_head)  #([32, 10, 40, 6]
        # (bs, n_head, n_q_seq, e_embd)
        #print(context.size()) # [32, 10, 240])
        output = self.linear(context)
        # print(output.size()) # [32, 10, 40]
        # print(output.size()) [32, 10, 40])
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob



""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.scale = 1 / (d_head ** 0.5)
    
    def forward(self, Q, K, V): #  ([32, 6, 40, 10]
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale) #([32, 6, 40, 40]
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V) #([32, 6, 40, 10]
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob



""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(40,160)
        self.linear2 = nn.Linear(160,40)
        self.active = F.gelu
        self.dropout = nn.Dropout(0.15)

    def forward(self, inputs):

        #print(inputs.size()) #[32, 10, 40]
        # (bs, d_ff, n_seq)
        
        output = self.linear1(inputs)
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.linear2(output)
        output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        # print(output.size()) # [32, 10, 40]

        return output




