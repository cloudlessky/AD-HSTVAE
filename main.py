# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,"/root/autodl-tmp/VAEAGG")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
from pathlib import Path
import json
import random
import datetime

from torch.utils.data import DataLoader, random_split, Subset
from sklearn.preprocessing import MinMaxScaler
from util.utils import *
from util.env import get_device, set_device
from util.preprocess import construct_data, construct_data_SMD
from TimeDataset import TimeDataset

import HSTVAE
from train import *
from evaluate import get_best_performance_data, get_full_err_scores

torch.cuda.current_device()
torch.cuda._initialized = True
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(linewidth=50)

def init_weights(m):  
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08) 

class Main():
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        NUM_TASKS = self.train_config['num_tasks']
        BATCH_SIZE = self.train_config['batch']
        LATENT_CODE_SIZE = self.train_config['LATENT_CODE_SIZE']
        T = self.train_config['slide_win']
        pred_win=self.train_config['pred_win']
        Spatio_n_hid = self.train_config['Spatio_n_hid']
        Spatio_n_head_dim = self.train_config['Spatio_n_head_dim']
        Spatio_head = self.train_config['Spatio_head']
        do_prob = self.train_config['do_prob']
        dec_in = self.train_config['dec_in']
        d_model = self.train_config['dim']
        alpha = self.train_config['alpha']
        beta = self.train_config['beta']

        dataset = self.env_config['dataset'] 
        if dataset == 'SMD':
            group_index = self.train_config['group_index']
            index = self.train_config['index']
            (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=True)
            train = torch.from_numpy(x_train).float()
            test = torch.from_numpy(x_test).float()
            labels = y_test if y_test is not None else None
            labels_list = labels.tolist()
           
            print('train.shape',train.shape)
            print('test.shape', test.shape)
            print('labels.shape', labels.shape)
        
            NUM_TASKS = train.shape[1]
            set_device(env_config['device'])  
            self.device = get_device()
            train_dataset_indata = construct_data_SMD(train, labels=0)  
            test_dataset_indata = construct_data_SMD(test, labels=labels_list)
            print('train_dataset_indata:', len(train_dataset_indata), len(train_dataset_indata[0]))
            print('test_dataset_indata:', len(test_dataset_indata), len(test_dataset_indata[0]))
            label1 = labels_list
        else: 
            train = pd.read_csv(f'../../data/{dataset}/train.csv', sep=',', index_col=0)
            test = pd.read_csv(f'../../data/{dataset}/test.csv', sep=',', index_col=0)
        
            if 'attack' in train.columns:
                train = train.drop(columns=['attack'])
            print('train.shape=', train.shape)  
            if 'P603' in train.columns:
                train = train.drop(columns=['P603'])  
            print('train.shape=', train.shape)  
            if 'P603' in test.columns:
                test = test.drop(columns=['P603'])  
            print('test.shape=', test.shape)  
            
            print('train.shape',train.shape)
            print('test.shape',test.shape)  

            set_device(env_config['device'])  
            self.device = get_device()
            label1 = test.attack.tolist()

            NUM_TASKS = train.shape[1]
            print('NUM_TASKS=', NUM_TASKS)

            train_dataset_indata = construct_data(train, labels=0)  
            test_dataset_indata = construct_data(test, labels=test.attack.tolist())
            
        train_config['num_tasks']=NUM_TASKS
        eval_config['num_tasks']=NUM_TASKS

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
            'pred_win': train_config['pred_win'],
        }

        train_dataset = TimeDataset(train_dataset_indata, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, mode='test', config=cfg)
       
        self.train_dataloader, self.val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])

        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0)

        
        enc_spatio = HSTVAE.Encoder_Spatio(T, Spatio_n_hid, Spatio_n_head_dim, Spatio_head,
                      do_prob, device=self.device).to(self.device)
    
        enc_temporal = HSTVAE.Encoder_Temporal(NUM_TASKS, LATENT_CODE_SIZE, factor=5, d_model=d_model, n_heads=8, e_layers=1, d_ff=d_model,
                 dropout=0, attn='full', activation='gelu',output_attention=False, distil=True,device=self.device).to(self.device)
        
        dec_rec = HSTVAE.Decoder_Rec(T, NUM_TASKS, LATENT_CODE_SIZE, factor=5, d_model=d_model, n_heads=8, d_layers=2, d_ff=d_model,
                dropout=0, attn='full',activation='gelu',mix=True, c_out=1, device=self.device).to(self.device)
        
        dec_pre = HSTVAE.Decoder_Pre(pred_win+1, NUM_TASKS, LATENT_CODE_SIZE,factor=5, d_model=d_model, n_heads=8, d_layers=2, d_ff=d_model,
                 dropout=0, attn='full',activation='gelu', mix=True, c_out=1, device=self.device).to(self.device)
        sampling_z = HSTVAE.Sampling_Temporal(NUM_TASKS, LATENT_CODE_SIZE, factor=5, d_model=d_model).to(self.device)
        gat = HSTVAE.GAT(input_size=train_config['dim'],
                        hidden_size=train_config['dim'],output_size=train_config['dim'],
                        num_of_task=NUM_TASKS,dropout=0,nheads=1,alpha=0.2).to(self.device)
       
        self.model = HSTVAE.VAE(enc_spatio, enc_temporal, sampling_z, gat, dec_pre, dec_rec,  T, LATENT_CODE_SIZE, dec_in, args.k,args.x0,d_model=d_model,topk_indices_ji=None)  # 用到了HSTVAE.py中定义的模型
       
        self.model.apply(init_weights)
        self.model.to(self.device)

    def run(self): 
        if len(self.env_config['load_model_path']) > 0:  
            model_save_path = self.env_config['load_model_path']
            print('model_save_path:', model_save_path)
        else:  
            print('**********************************train***********************************')
            begin_train = datetime.now()
            model_save_path = train(self.model, self.device, config = train_config, train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader)
            end_train = datetime.now()
            train_time = (end_train - begin_train) 
            f = open('./{}/train_time.txt'.format(train_config['loss_path']), 'a')
            print(train_time.total_seconds(), file=f)
            f.close()
        # test
        f = open('./{}/f1.txt'.format(train_config['loss_path']), 'a')
        print('********model_save_path:', model_save_path, file=f)
        f.close()
        begin_infer = datetime.now()
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)
        print('**************************************test***********************************')
        _, self.test_result, self.test_future_result, self.test_history_result = test(best_model, self.test_dataloader, train_config, self.device)
        print('**************************************score***********************************')
        self.get_score(self.test_result, self.test_future_result, self.test_history_result, self.train_config['alpha'], self.train_config['beta'])
        end_infer = datetime.now()
        infer_time = (end_infer - begin_infer) 
        f = open('./{}/infer_time.txt'.format(train_config['loss_path']), 'a')
        print(infer_time.total_seconds(), file=f)
        f.close()

    def get_score(self, test_result, test_future_result, test_history_result, alpha, beta):
        np_test_result = np.array(test_result)
        true_labels = np_test_result[2, :, 0].tolist()  
        test_scores_nowpred, test_scores_abs, test_future_scores, test_history_scores = get_full_err_scores(test_result, test_future_result, test_history_result)
        best_res = get_best_performance_data(test_scores_nowpred, test_scores_abs, test_future_scores, test_history_scores, true_labels, config=eval_config)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.2):
        dataset_len = int(len(train_dataset)) 
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)  
        f = open('./{}/f1.txt'.format(train_config['loss_path']), 'a')
        print('***val_start_index= ',val_start_index, 'val_use_len= ', val_use_len, file=f)
        f.close()

        indices = torch.arange(dataset_len)  
        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len:]])  
        train_subset = Subset(train_dataset, train_sub_indices)  
        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]  
        val_subset = Subset(train_dataset, val_sub_indices) 

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    

if __name__ == "__main__":
    starttime = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', help='batch size', type=int, default=512)
    parser.add_argument('-epoch', help='train epoch', type=int, default=200)
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='swat')
    parser.add_argument('-group_index', help='group_index', type=int, default=0)
    parser.add_argument('-index', help='index', type=int, default=0)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('-pred_win', help='pred_win', type=int, default=5)
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-num_tasks', help='num_tasks', type=int, default=50) 
    parser.add_argument('-LATENT_CODE_SIZE', help='LATENT_CODE_SIZE', type=int, default=3)
    parser.add_argument('-prior', help='prior', type=np.array, default=np.array([0.9,0.1])) # swat-0.1; wadi-0.2; smd-0.13
    parser.add_argument('-Spatio_head', help='Spatio_head', type=int, default=1)
    parser.add_argument('-kl_coff', help='kl_coff', type=float, default=50)
    parser.add_argument('-pred_now_coff', help='pred_now_coff', type=int, default=3)
    parser.add_argument('-pred_future_coff', help='pred_future_coff', type=int, default=2)
    parser.add_argument('-rec_coff', help='rec_coff', type=int, default=1)
    parser.add_argument('-rec_now_coff', help='rec_now_coff', type=int, default=1)

    parser.add_argument('-dec_in', help='dec_in', type=int, default=1)
    parser.add_argument('-dim', help='dimension', type=int, default=64) 
    parser.add_argument('-down', help='down', type=int, default=10)
    parser.add_argument('-topk', help='topk', type=int, default=49)
    parser.add_argument('-lr', help='lr', type=float, default=0.001)
    parser.add_argument('-patience', help='patience', type=int, default=5)

    parser.add_argument('-ano_file_path', help='ano_file_path', type=str, default='../data/swat/SWAT_Time.csv')
    parser.add_argument('-loss_path', help='loss_path', type=str, default='loss_swat')
    parser.add_argument('-alpha', help='alpha', type=float, default=0.05)
    parser.add_argument('-beta', help='beta', type=float, default=0.05)
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)

    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=16)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-n_hid', help='n_hid', type=int, default=64)
    parser.add_argument('-do_prob', help='do_prob', type=int, default=0)
    parser.add_argument('-Spatio_n_hid', help='Spatio_n_hid', type=int, default=64)
    parser.add_argument('-Spatio_n_head_dim', help='Spatio_n_head_dim', type=int, default=32)
    parser.add_argument('-temperature', help='temperature', type=int, default=0.5)
    parser.add_argument('-GRU_n_dim', help='GRU_n_dim', type=int, default=64)
    parser.add_argument('-max_diffusion_step', help='max_diffusion_step', type=int, default=2)
    parser.add_argument('-num_rnn_layers', help='num_rnn_layers', type=int, default=1)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')

    parser.add_argument('-k', help='--k', type=float, default=0.0025)
    parser.add_argument('-x0', help='--x0', type=int, default=6000)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'lr':args.lr,
        'patience':args.patience,
        'slide_win': args.slide_win,
        'pred_win': args.pred_win,
        'dim': args.dim,
        'dataset': args.dataset,
        'ano_file_path':args.ano_file_path,
        'loss_path':args.loss_path,
        'slide_stride': args.slide_stride,
        'kl_coff': args.kl_coff ,
        # 'spatiokl_coff': args.spatiokl_coff ,
        'pred_now_coff': args.pred_now_coff,
        'rec_now_coff': args.rec_now_coff,
        'pred_future_coff': args.pred_future_coff,
        'rec_coff': args.rec_coff,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'num_tasks': args.num_tasks,
        'LATENT_CODE_SIZE': args.LATENT_CODE_SIZE,
        'n_hid': args.n_hid,
        'do_prob': args.do_prob,
        'Spatio_n_hid': args.Spatio_n_hid,
        'Spatio_n_head_dim': args.Spatio_n_head_dim,
        'Spatio_head': args.Spatio_head,
        'prior': args.prior,
        'temperature': args.temperature,
        'GRU_n_dim': args.GRU_n_dim,
        'max_diffusion_step': args.max_diffusion_step,
        'num_rnn_layers': args.num_rnn_layers,
        'dec_in':args.dec_in,
        'alpha': args.alpha,
        'beta': args.beta,
        'down': args.down,
        'group_index': args.group_index,
        'index': args.index,
        'k': args.k,
        'x0': args.x0,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'loss_path':args.loss_path,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'group_index': args.group_index,
        'index': args.index,
        'topk': args.topk,
        'LATENT_CODE_SIZE': args.LATENT_CODE_SIZE,
    }

    eval_config = {
        'slide_win': args.slide_win,
        'pred_win': args.pred_win,
        'down': args.down,
        'num_tasks': args.num_tasks,
        'dataset': args.dataset,
        'ano_file_path':args.ano_file_path,
        'loss_path':args.loss_path,
        'group_index': args.group_index,
        'index': args.index,
        'topk': args.topk,
        'LATENT_CODE_SIZE': args.LATENT_CODE_SIZE,
        'alpha': args.alpha,
        'beta': args.beta,
    }


    main = Main(train_config, env_config, debug=False)
    main.run()
    endtime = datetime.now()
    
    t = (endtime - starttime).seconds
    print('***The total time is ', t)
    f = open('./{}/total_time.txt'.format(train_config['loss_path']), 'a')
    print('t:', t, file=f)
    f.close()
