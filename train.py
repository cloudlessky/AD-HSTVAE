import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import random
from util.time import *
from util.env import *
from util.data import *
from lib.utils import *
import torch.utils.data

from torch import optim
from EarlyStopping import EarlyStopping
from torch.autograd import Variable

torch.set_default_tensor_type(torch.FloatTensor)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=50)

criterion = nn.MSELoss()

def epoch_time(start_time, end_time):  
    elapsed_time = end_time - start_time 
    elapsed_mins = int(elapsed_time / 60)  
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60)) 
    return elapsed_mins, elapsed_secs


def calculate_kl_loss(z_mean, z_log_sigma): 
    """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
    temp = 1.0 + 2 * z_log_sigma - z_mean ** 2 - torch.exp(2 * z_log_sigma)
    return -0.5 * torch.sum(temp, 1)  

def test(model, testloader, config, device):
    model.eval()
    test_loss_list = []
    pred_now_coff=config['pred_now_coff']
    pred_future_coff=config['pred_future_coff']
    rec_coff=config['rec_coff']
    kl_coff=0
    Bernoulli_prior = torch.FloatTensor(config['prior'])
    Bernoulli_prior = torch.unsqueeze(Bernoulli_prior, 0)
    Bernoulli_prior = torch.unsqueeze(Bernoulli_prior, 0)
    Bernoulli_prior = Variable(Bernoulli_prior)
    Bernoulli_prior = Bernoulli_prior.to(device)
    

    batch_i=0
    for batch_input, batch_target, attack_labels in testloader:
        with torch.no_grad():
            batch_input=batch_input.to(device)
            batch_target=batch_target.to(device)
            rec, pred, mu, sigma, prob = model(batch_input, batch_target, batch_i, config, 0, 0, device)
            kl_loss = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
            recon_loss = criterion(rec.float(), batch_input.float())
            pred_now_loss = criterion(pred[:,:,0].float(), batch_target[:,:,0].float())
            pred_future_loss = criterion(pred[:,:,1:].float(), batch_target[:,:,1:].float())
            batch_loss = rec_coff*recon_loss + kl_coff * kl_loss + pred_now_coff*pred_now_loss +pred_future_coff*pred_future_loss
            spatio_kl_loss = kl_categorical(torch.mean(prob, 1), Bernoulli_prior, torch.var(prob, 1), 1).float()
            batch_loss += spatio_kl_loss
            batch_loss=batch_loss.float()
            
            batch_i+=1
            pred_t = pred[:, :, 0]
            y_pred = batch_target[:,:,0]
            pred_future = pred[:, :, 1:]
            y_pred_future = batch_target[:, :, 1:]
            recon_history = rec
            y_recon_history = batch_input
            labels = attack_labels.unsqueeze(1).repeat(1, pred.shape[1])

            if batch_i == 1:
                t_test_pre = pred_t
                t_test_trg = y_pred

                t_test_pre_future = pred_future
                t_test_trg_future = y_pred_future

                t_test_rec_history = recon_history
                t_test_trg_history = y_recon_history
                t_test_labels = labels
            else:
                t_test_pre = torch.cat((t_test_pre, pred_t), dim=0)
                t_test_trg = torch.cat((t_test_trg, y_pred), dim=0)
                t_test_pre_future = torch.cat((t_test_pre_future, pred_future), dim=0)
                t_test_trg_future = torch.cat((t_test_trg_future, y_pred_future), dim=0)
                t_test_rec_history = torch.cat((t_test_rec_history, recon_history), dim=0)
                t_test_trg_history = torch.cat((t_test_trg_history, y_recon_history), dim=0)
                t_test_labels = torch.cat((t_test_labels, labels), dim=0)
        test_loss_list.append(batch_loss.item()) 
        
    test_pre_list = t_test_pre.tolist()
    test_trg_list = t_test_trg.tolist()
    test_pre_future = t_test_pre_future.tolist()
    test_trg_future = t_test_trg_future.tolist()
    test_rec_history = t_test_rec_history.tolist()
    test_trg_history = t_test_trg_history.tolist()
    test_labels_list = t_test_labels.tolist()
    avg_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_loss,[test_pre_list, test_trg_list, test_labels_list],[test_pre_future, test_trg_future],[test_rec_history, test_trg_history]


def val(model,testloader,config, epoch, device):
    model.eval()
    kl_turn=config['kl_coff']
    pred_now_coff=config['pred_now_coff']
    pred_future_coff=config['pred_future_coff']
    rec_coff=config['rec_coff']

    Bernoulli_prior = torch.FloatTensor(config['prior'])
    Bernoulli_prior = torch.unsqueeze(Bernoulli_prior, 0)
    Bernoulli_prior = torch.unsqueeze(Bernoulli_prior, 0)
    Bernoulli_prior = Variable(Bernoulli_prior)
    Bernoulli_prior = Bernoulli_prior.to(device)

    kl_coff=0
    acu_loss = 0
    batch_i=0
    for batch_input, batch_target, _ in testloader:
        batch_i += 1
        with torch.no_grad():
            batch_input=batch_input.to(device)
            batch_target=batch_target.to(device)
            rec,pred,mu, sigma, prob = model(batch_input, batch_target, batch_i, config, 1, 0, device)

            kl_loss = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
            recon_loss = criterion(rec.float(), batch_input.float())
            pred_now_loss = criterion(pred[:,:,0].float(), batch_target[:,:,0].float())
            pred_future_loss = criterion(pred[:,:,1:].float(), batch_target[:,:,1:].float())
            batch_loss = rec_coff*recon_loss + kl_coff * kl_loss + pred_now_coff*pred_now_loss + pred_future_coff*pred_future_loss
            spatio_kl_loss = kl_categorical(torch.mean(prob, 1), Bernoulli_prior, torch.var(prob, 1), 1).float()
            batch_loss += spatio_kl_loss
            batch_loss=batch_loss.float()
        acu_loss+=batch_loss
    avg_loss = acu_loss / batch_i
    return avg_loss

def train(model = None, device=None, config={}, train_dataloader=None, val_dataloader=None):
    N_EPOCHS = config['epoch']
    dataset = config['dataset']
    group_index = config['group_index']
    index = config['index']
    kl_turn=config['kl_coff']
    pred_now_coff=config['pred_now_coff']
    pred_future_coff=config['pred_future_coff']
    rec_coff=config['rec_coff']
    loss_path=config['loss_path']
    LR=config['lr']
    patience=config['patience']

    train_loader = train_dataloader
    valid_loader = val_dataloader

    optimizer = optim.Adam(model.parameters(), lr=LR)
    es = EarlyStopping(patience=patience)

    Bernoulli_prior = torch.FloatTensor(config['prior'])
    Bernoulli_prior = torch.unsqueeze(Bernoulli_prior, 0)
    Bernoulli_prior = torch.unsqueeze(Bernoulli_prior, 0)
    Bernoulli_prior = Variable(Bernoulli_prior)
    Bernoulli_prior = Bernoulli_prior.to(device)

    min_loss = 1e+8
        
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        batch_losses = 0
        batch_i = 0
        if epoch<kl_turn:
            kl_coff=0
        else:
            kl_coff=(1e-5)*np.power(10, epoch-kl_turn)
            if kl_coff>1:
                kl_coff=1

        for all_input, all_target_seq, _ in train_loader:
            optimizer.zero_grad()

            all_input=all_input.to(device)
            all_target_seq=all_target_seq.to(device)
            NUM_TASKS = all_input.shape[1]
            rec, pred, mu, sigma, prob = model(all_input, all_target_seq, batch_i,config,1,epoch,device)
            
            kl_loss = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
            recon_loss = criterion(rec.float(), all_input.float())
            pred_now_loss = criterion(pred[:,:,0].float(), all_target_seq[:,:,0].float())
            pred_future_loss = criterion(pred[:,:,1:].float(), all_target_seq[:,:,1:].float())
            batch_loss = rec_coff*recon_loss + kl_coff * kl_loss + pred_now_coff*pred_now_loss + pred_future_coff*pred_future_loss
            spatio_kl_loss = kl_categorical(torch.mean(prob, 1), Bernoulli_prior, torch.var(prob, 1), 1).float()
            batch_loss += spatio_kl_loss
            batch_loss = batch_loss.float()
            
            batch_i += 1
            batch_losses += batch_loss
            batch_loss = batch_loss*NUM_TASKS
            batch_loss.backward()
            optimizer.step()
        epoch_loss = batch_losses / batch_i

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s'.format(epoch=epoch, epoch_mins=epoch_mins, epoch_secs=epoch_secs))
        print('\tTrain Loss: {train_loss:.3f}'.format(train_loss = epoch_loss))
        f = open('./{}/train_loss.txt'.format(config['loss_path']), 'a')
        print('epoch:', epoch, 'group-index', group_index, '-', index,' train_loss:', epoch_loss, 'train time', epoch_mins,epoch_secs, file=f)
        f.close()
       
        valid_loss = val(model, valid_loader, config, epoch, device)
        valid_loss_cpu=valid_loss.cpu()

        if es.step(valid_loss_cpu):
            print("Early Stopping")
            break

        if dataset == 'swat' or dataset == 'wadi':
            save_path = f'./pretrained/{dataset}/best_{dataset}-epoch{epoch}.pt'
        elif dataset == 'SMD':
            save_path = f'./pretrained/{dataset}/best_{dataset}_group{group_index}_index{index}.pt'
            
        
        if valid_loss < min_loss: 
            min_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            final_save_path = save_path

        print('\t Val. Loss: {valid_loss}'.format(valid_loss=valid_loss))
        f = open('./{}/valid_loss.txt'.format(config['loss_path']), 'a')
        print('epoch:', epoch, 'group-index', group_index, '-', index, ' valid_loss:', valid_loss, 'min_loss', min_loss, file=f)
        f.close()

    return final_save_path



