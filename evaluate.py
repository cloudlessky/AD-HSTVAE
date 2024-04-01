import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.data import *
from sklearn.metrics import roc_auc_score

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=50)

def get_full_err_scores(test_result, test_future_result, test_history_result):
    np_test_result = np.array(test_result,dtype=object)  
    np_test_future_result = np.array(test_future_result,dtype=object) 
    np_test_history_result = np.array(test_history_result,dtype=object) 
    
    all_scores_pred = None
    all_scores_abs = None
    all_scores_future = None
    all_scores_history = None
    feature_num = np_test_result.shape[-1]  

    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]  
        test_recon_list = np_test_history_result[:,:,i,-1]
        test_future_list = np_test_future_result[:,:,i,:]  
        test_history_list = np_test_history_result[:,:,i,:-1]
       
        scores_curpred = get_err_scores(test_re_list)
        scores_current_recon = get_err_scores(test_recon_list)
        scores_current_abs = get_abs_err_scores(test_re_list)
      
        scores_pred = scores_curpred
        scores_currecon = scores_current_recon
        scores_now = scores_curpred + scores_currecon
        scores_future = get_max_err_scores(test_future_list)
        scores_history = get_max_err_scores(test_history_list)
        
        if all_scores_pred is None:
            all_scores_pred = scores_pred  
        else:
            all_scores_pred = np.vstack((  
                all_scores_pred,
                scores_pred
            ))

        if all_scores_abs is None:
            all_scores_abs = scores_current_abs 
        else:
            all_scores_abs = np.vstack((  
                all_scores_abs,
                scores_current_abs
            ))

        if all_scores_future is None:
            all_scores_future = scores_future
        else:
            all_scores_future = np.vstack((
                all_scores_future,
                scores_future
            ))

        if all_scores_history is None:
            all_scores_history = scores_history
        else:
            all_scores_history = np.vstack((
                all_scores_history,
                scores_history
            ))
    return all_scores_pred, all_scores_abs, all_scores_future, all_scores_history


def get_max_err_scores(test_res): 
    test_predict, test_gt = test_res
    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    max_test_delta = test_delta.max(axis=1)  

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    epsilon = 1e-2
    err_scores = (max_test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)  
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])
    return err_scores  

def get_err_scores(test_res):
    test_predict, test_gt = test_res
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    test_delta = torch.zeros(test_predict.shape)
    for i in range(len(test_predict)):
        test_delta[i] = np.sqrt((test_predict[i] - test_gt[i]) ** 2) 
    q75, q25 = np.percentile(test_delta, [75, 25])
    iqr = q75 - q25
    median = np.median(test_delta)
    a_score = (test_delta - median) / (1 + iqr)
    return a_score

def get_abs_err_scores(test_res):
    test_predict, test_gt = test_res
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    test_delta = np.abs(np.subtract(     
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)  
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])
    return smoothed_err_scores

def get_best_performance_data(total_err_scores, total_err_scores_abs_delta, future_scores, history_scores, gt_labels, config={}):
    group_index = config['group_index']
    index = config['index']
    alpha = config['alpha']
    beta = config['beta']
    topk = config['topk']

    total_features = total_err_scores.shape[0] 
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]  
    total_topk_err_scores = np.mean(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    topk_indices_future = np.argpartition(future_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:] 
    future_topk_err_scores = np.mean(np.take_along_axis(future_scores, topk_indices_future, axis=0), axis=0)

    topk_indices_history = np.argpartition(history_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]  
    history_topk_err_scores = np.mean(np.take_along_axis(history_scores, topk_indices_history, axis=0), axis=0)
    
    topk_indices = np.argpartition(total_err_scores_abs_delta, range(total_features - topk - 1, total_features),axis=0)[-topk:]  # topk=1
    total_topk_err_scores_abs_delta = np.mean(np.take_along_axis(total_err_scores_abs_delta, topk_indices, axis=0),axis=0)

    delta_erro = np.zeros(len(total_topk_err_scores_abs_delta))
    delta_erro[0] = total_topk_err_scores_abs_delta[0]
    for i in range(1, len(total_topk_err_scores_abs_delta)):
        delta_erro[i] = total_topk_err_scores_abs_delta[i] - total_topk_err_scores_abs_delta[i - 1]

    auc_mse_pre = roc_auc_score(gt_labels, total_topk_err_scores) 
    auc_his = roc_auc_score(gt_labels, history_topk_err_scores)  
    auc_fut = roc_auc_score(gt_labels, future_topk_err_scores)  
    auc_abs = roc_auc_score(gt_labels, total_topk_err_scores_abs_delta)  
    auc_delta = roc_auc_score(gt_labels, delta_erro)  
    print('auc_mse_pre, auc_his, auc_fut, auc_abs, auc_delta ', auc_mse_pre, auc_his, auc_fut, auc_abs, auc_delta)

    f = open('./{}/f1.txt'.format(config['loss_path']), 'a')
    print('******group_index, index', group_index, '-', index, file=f)
    print('dataset', config['dataset'], 'slide_win', config['slide_win'], 'pred_win',config['pred_win'],'down',
    config['down'],'topk',config['topk'], 'LATENT_CODE_SIZE',config['LATENT_CODE_SIZE'], file=f)
    print('auc_mse_pre, auc_his, auc_fut, auc_abs, auc_delta', auc_mse_pre, auc_his, auc_fut, auc_abs, auc_delta, file=f)
    f.close()

    gt_labels = np.array(gt_labels)
    pred_now_bf_eval = bf_search(total_topk_err_scores, gt_labels, start=0.01, end=2, step_num=100, verbose=False)
    print('pred_now_bf_eval=', pred_now_bf_eval)
    
    nhf_score=[]
    for i in range(len(total_topk_err_scores)):
        nhf_score.append((1-alpha-beta)*total_topk_err_scores[i] + alpha*history_topk_err_scores[i] + beta*future_topk_err_scores[i])
    nhf_bf_eval = bf_search(nhf_score, gt_labels, start=0.01, end=2, step_num=100, verbose=False)
    print('nhf_bf_eval=', nhf_bf_eval)
    auc_nhf = roc_auc_score(gt_labels, nhf_score)
    print('auc_nhf=', auc_nhf) 

    f = open('./{}/f1.txt'.format(config['loss_path']), 'a')
    print('auc_nhf=', auc_nhf, file=f)
    print('pred_now_bf_eval=', pred_now_bf_eval, file=f)
    print('nhf_bf_eval=', nhf_bf_eval, file=f)
    f.close()

    return pred_now_bf_eval
