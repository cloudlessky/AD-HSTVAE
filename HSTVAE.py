from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, transforms
import random
from lib.utils import *
import networkx as nx
import matplotlib.pyplot as plt

from util.masking import TriangularCausalMask, ProbMask
from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, Sampling, SampLayer,RecLinear,LinearLayer
from decoder import DecoderR, DecoderP, DecoderRecLayer, DecoderPreLayer
from attn import FullAttention, ProbAttention, AttentionLayer_Rec, AttentionLayer_Pre_Cross, AttentionLayer
from embed import DataEmbedding,LstmEmbedding
from sklearn import manifold

SEED = 2020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

num = 0

class Enc_linear(nn.Module):
    """Linear for mu and sigma."""
    def __init__(self, n_in, n_hid, n_out):
        super(Enc_linear, self).__init__()
        super(Enc_linear, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = torch.sigmoid(self.fc1(inputs))
        x = torch.sigmoid(self.fc2(x))
        return self.batch_norm(x)

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class MLP2(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_out, do_prob=0.):
        super(MLP2, self).__init__()
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = self.fc1(inputs)
        return x

def norm(t):
    return t / t.norm(dim=1, keepdim=True)

def cos_sim(v1, v2):
    v1 = norm(v1)
    v2 = norm(v2)
    return v1 @ v2.t()

class Encoder_Temporal(nn.Module):
    def __init__(self, num_tasks, latent_dim, factor=5, d_model=64, n_heads=8, e_layers=1, d_ff=64,
                 dropout=0.0, attn='full', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(Encoder_Temporal, self).__init__()
        self.attn = attn
        self.output_attention = output_attention
        self.Nz = latent_dim

        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder_temporal = nn.ModuleList([Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for i in range(num_tasks)])
       
    def forward(self, num_of_task, inp_enc, enc_self_mask=None):
        enc_out,_ = self.encoder_temporal[num_of_task](inp_enc, enc_self_mask)
        return enc_out


class Sampling_Temporal(nn.Module):
    def __init__(self, num_tasks, latent_dim, factor=5, d_model=64,
                 device=torch.device('cuda:0')):
        super(Sampling_Temporal, self).__init__()
        self.Nz = latent_dim
        
        self.mu = nn.ModuleList([Enc_linear(d_model, int(d_model/2), self.Nz) for i in range(num_tasks)])
        self.sigma = nn.ModuleList([Enc_linear(d_model, int(d_model/2), self.Nz) for i in range(num_tasks)])

    def forward(self, num_of_task, enc_out):
        mu = self.mu[num_of_task](enc_out).cuda()  
        sigma_hat = self.sigma[num_of_task](enc_out).cuda() 
        F = nn.Tanh()
        sigma_hat = F(sigma_hat).to(device)
        z = self.z_sample(mu, sigma_hat)
        return z, mu, sigma_hat
    
    def z_sample(self, mu, logvar):
        epsilon = torch.randn(mu.shape).to(device)
        return mu + torch.exp(logvar * 0.5) * epsilon

class Decoder_Rec(nn.Module):
    def __init__(self, win_len, num_tasks, latent_dim, factor=5, d_model=64, n_heads=8, d_layers=1, d_ff=64,
                dropout=0.0, attn='full',activation='gelu',mix=True, c_out=1,
                device=torch.device('cuda:0')):
        super(Decoder_Rec, self).__init__()

        self.fc_in = nn.ModuleList([nn.Linear(1, win_len, bias=True) for i in range(num_tasks)])
        self.fc_in2 = nn.ModuleList([nn.Linear(latent_dim, d_model, bias=True) for i in range(num_tasks)])
        self.activation = F.relu

        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.decoder_rec = nn.ModuleList([DecoderR(
            [
                DecoderRecLayer(
                    AttentionLayer_Rec(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, d_model, n_heads, mix=mix),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for i in range(num_tasks)])

        self.fc_out = nn.ModuleList([nn.Linear(d_model, c_out, bias=True) for i in range(num_tasks)])

    def forward(self, num_of_task, z, dec_self_mask=None):
        z=z.permute(0,2,1)
        z1 = self.fc_in[num_of_task](z)
        z1=z1.permute(0,2,1)
        z1 = self.fc_in2[num_of_task](z1)
        z1=self.activation(z1)
        dec_rec_out = self.decoder_rec[num_of_task](z1, x_mask=dec_self_mask)
        recon_out = self.fc_out[num_of_task](dec_rec_out)
        return recon_out

class Decoder_Pre(nn.Module):
    def __init__(self, pred_len, num_tasks, latent_dim, factor=5, d_model=64, n_heads=8, d_layers=1, d_ff=64,
                 dropout=0.0, attn='full',activation='gelu', mix=True, c_out=1,
                 device=torch.device('cuda:0')):
        super(Decoder_Pre, self).__init__()
        self.fc_in = nn.ModuleList([nn.Linear(1, pred_len, bias=True) for i in range(num_tasks)])
        self.fc_in2 = nn.ModuleList([nn.Linear(latent_dim, d_model, bias=True) for i in range(num_tasks)])
        self.activation = F.relu

        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.decoder_pre = nn.ModuleList([DecoderP(
            [
                DecoderPreLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer_Pre_Cross(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for i in range(num_tasks)])

        self.fc_out = nn.ModuleList([nn.Linear(d_model, c_out, bias=True) for i in range(num_tasks)])

    def forward(self, num_of_task, dec_out, z, dec_self_mask=None, dec_enc_mask=None):
        z=z.permute(0,2,1)
        z1 = self.fc_in[num_of_task](z)
        z1=z1.permute(0,2,1)
        z1 = self.fc_in2[num_of_task](z1)
        z1=self.activation(z1)
        
        dec_rec_out = self.decoder_pre[num_of_task](dec_out, z1, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        pred_out = self.fc_out[num_of_task](dec_rec_out)
        return pred_out

class Encoder_Spatio(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, do_prob=0., device=None):  
        super(Encoder_Spatio, self).__init__()
        self.n_hid = n_hid
        self.head = head
        self.n_in = n_in
        self.n_head_dim = n_head_dim
        self.device = device

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)

        self.n_hid2 = n_head_dim
        self.mlp2 = MLP2(n_head_dim*2, 1, do_prob)
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs): 
        X = self.mlp1(inputs)
        Xq = self.Wq(X) 
        Xk = self.Wk(X)
        B, N, n_hid = Xq.shape
        Xq = Xq.view(B, N, self.head, self.n_head_dim) 
        Xk = Xk.view(B, N, self.head, self.n_head_dim)
        Xq = Xq.permute(0, 2, 1, 3).squeeze()  
        Xk = Xk.permute(0, 2, 1, 3).squeeze()
        XQ = Xq.unsqueeze(2)
        XQ = XQ.expand(-1,-1,Xk.shape[1],-1)
        NQ = XQ.reshape(Xk.shape[0],Xk.shape[1]*Xk.shape[1],Xk.shape[2])
        NK = Xk.repeat(1,Xk.shape[1],1)

        NN_qk = torch.cat((NK,NQ),-1)
        probs = self.mlp2(NN_qk)
        probs = probs.reshape(Xk.shape[0],Xk.shape[1],Xk.shape[1])
        return probs   

class GATLayer(nn.Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, num_of_task, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(num_of_task, 2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
        self.beta = 0.3

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj,batch_i):
        N, C = h.size()
        Wh = torch.matmul(h, self.w)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, C), Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.output_feature) 
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) 
        zero_vec = -9e40 * torch.ones_like(e)
        
        attention = torch.where(adj > 0, e, zero_vec)  
        attention = F.softmax(attention, dim=1)  
        h_prime = torch.mm(attention, Wh)  
        out = F.elu(h_prime + self.beta * Wh)
        return out

class GAT(nn.Module):
    """GAT模型"""
    def __init__(self, input_size, hidden_size, output_size, num_of_task, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, num_of_task, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, num_of_task, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj,batch_i):
        x = torch.cat([att(x, adj,batch_i) for att in self.attention], dim=1)
        return x

def kl_anneal_function(step, k, x0):
    return float(1/(1+np.exp(-k*(step-x0))))
       

class MTVAE(nn.Module):
    def __init__(self, enc_spatio, enc_temporal, sampling_z, GAT, dec_rec, dec_pre, enc_in, dec_in, k, x0,d_model=64, topk_indices_ji=None, dropout=0.2, embed='fixed', freq='t'):
        super(MTVAE, self).__init__()

        self.enc_spatio = enc_spatio
        self.enc_temporal = enc_temporal
        self.dec_rec = dec_rec
        self.dec_pre = dec_pre
        self.sampling_z = sampling_z
        self.topk_indices_ji = topk_indices_ji
        self.GAT = GAT
        self.head = 1
        self.temperature = 0.5
        self.k = k
        self.x0 = x0

       
        self.enc_embedding = LstmEmbedding(1, d_model)#
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

    def forward(self, all_input, all_target, batch_i, config, flag, step, device):
        NUM_OF_TASK = all_input.shape[1]
        batch_size = all_input.shape[0]
        all_input = all_input.to(device).float()
        all_target = all_target.to(device).float()

        probs = self.enc_spatio(all_input).to(device)  
        probs = probs.reshape(probs.shape[0], probs.shape[1] * probs.shape[1], 1)
        full_1 = torch.ones(probs.shape).to(device)
        probs_0 = full_1 - probs
        prob_cat = torch.cat((probs,probs_0),2).to(device)

        probs = prob_cat.permute(0,2,1)
        probs = probs.reshape(batch_size,2,NUM_OF_TASK,NUM_OF_TASK)
        mask_loc = torch.eye(NUM_OF_TASK, dtype=bool).to(device)
        probs_reshaped = probs.masked_select(~mask_loc).view(batch_size, 2, NUM_OF_TASK * (NUM_OF_TASK - 1)).to(device)
        probs_reshaped = probs_reshaped.permute(0, 2, 1)
        prob = F.softmax(probs_reshaped, -1)
        
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(device)

        adj_list = torch.ones(2, batch_size, NUM_OF_TASK, NUM_OF_TASK).to(device)
        mask = ~torch.eye(NUM_OF_TASK, dtype=bool).unsqueeze(0).unsqueeze(0).to(device)
        mask = mask.repeat(2, batch_size, 1, 1).to(device)
        adj_list[mask] = edges.permute(2, 0, 1).flatten()
        adj = adj_list[1,:,:,:]


        for num_of_task in range(NUM_OF_TASK):
            src = all_input[:, num_of_task, :]
            src = src.unsqueeze(2).to(device)
            src = src.float()
            enc_inp = self.enc_embedding(src)
            
            enc_out = self.enc_temporal(num_of_task, enc_inp, enc_self_mask=None)
            enc_out = enc_out[:,-1,:]

            enc_out = enc_out.unsqueeze(1)
            if num_of_task == 0:
                enc_out_cat = enc_out
            else:
                enc_out_cat = torch.cat((enc_out_cat, enc_out), dim=1)
        enc_out_update = torch.zeros_like(enc_out_cat)、

        for i in range(enc_out_cat.shape[0]):
            enc_out_update[i, :] = self.GAT(enc_out_cat[i, :], adj[i, :],batch_i).float().to(device)
        
        for num_of_task in range(NUM_OF_TASK):
            src = all_input[:, num_of_task, :].to(device)    
            trg_now = all_target[:, num_of_task,0].to(device)  
            trg_future = all_target[:, num_of_task, 1:].to(device)  

            enc_out1 = enc_out_update[:, num_of_task, :]
            enc_out1 = enc_out1.unsqueeze(1)

            z,mu,sigma = self.sampling_z(num_of_task, enc_out1)
            
            rec_output = self.dec_rec(num_of_task, z, dec_self_mask=None)
            
            rec_output = rec_output.squeeze()
            src = src.squeeze()
            batch_size = src.shape[0]
            token = torch.rand(batch_size, 1,1).to(device)
            dec_pre_inp = torch.zeros_like(trg_future.unsqueeze(2))
            dec_inp_init = torch.cat([token, dec_pre_inp], dim=1)
            dec_inp = self.dec_embedding(dec_inp_init)
            pred_output = self.dec_pre(num_of_task, dec_inp, z, dec_self_mask=None, dec_enc_mask=None)
            
            rec_output_return = rec_output.unsqueeze(1)
            pred_output_return = pred_output.squeeze(2).unsqueeze(1)  
            if num_of_task == 0:
                rec_out_cat = rec_output_return
                pre_out_cat = pred_output_return
                mu_cat = mu
                sigma_cat = sigma
            else:
                rec_out_cat = torch.cat((rec_out_cat, rec_output_return), dim=1)  
                pre_out_cat = torch.cat((pre_out_cat, pred_output_return), dim=1) 
                mu_cat = torch.cat((mu_cat, mu), dim=1)
                sigma_cat = torch.cat((sigma_cat, sigma), dim=1)
        return rec_out_cat, pre_out_cat, mu_cat, sigma_cat, prob



