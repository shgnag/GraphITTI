import os
import csv
import json
import python_speech_features
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr

def rank_loss_cal(batch_y_out,batch_order_y_true, device):
    num_nodes = 4
    batch_loss = []
    rank = []
    ar1 = []
    ar2 = []
    for i in range(len(batch_y_out)):
        y_out = batch_y_out[i,:]
        order_y_true = batch_order_y_true[i,:]
        model_size = len(y_out)
        y_out = y_out.reshape((model_size))
        order_y_true = order_y_true.reshape((model_size))

        sample_num = num_nodes*4
        # sample_num simply decides how many possibilities of ind1 and 2 to be generated for comparision
        ind_1 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
        ind_2 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)

        input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
        input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)

        # Label acc to ground truth ranks
        rank_measure = torch.zeros((input_arr2.shape[0]), dtype=torch.int)
        for y_i in range(len(input_arr1)):
            
            l1 = np.where(y_out.cpu().detach().numpy()== input_arr1[y_i].cpu().detach().numpy())[0]
            l2 = np.where(y_out.cpu().detach().numpy()== input_arr2[y_i].cpu().detach().numpy())[0]
            
            if len(l2) >1 or len(l1) >1:
                l2 = l2[0]
                l1 = l1[0]
            if l1==l2:
                rank_measure[y_i] = 0
            elif l1> l2:
                rank_measure[y_i] = 1 # if 0 is most otherwise y=1
            elif l1<l2:
                rank_measure[y_i] = -1
        rank_measure = rank_measure.to(device)

        # https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
        loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1,input_arr2,rank_measure)
        batch_loss.append(loss_rank)

    batch_loss = torch.stack(batch_loss)
    return torch.sum(batch_loss)


def ranking_correlation(y_out,true_val):
    
    model_size = len(y_out)


    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()

    batch_kt = []
    for i in range(len(true_arr)): # calculating score for each graph as ranks between 0-3 for every graph

        kt,_ = kendalltau(predict_arr[i,:],true_arr[i,:],nan_policy = 'omit')
        if np.isnan(kt):
            kt = np.nan_to_num(kt)

        batch_kt.append(kt)

    return batch_kt

def SPC_ranking_correlation(y_out,true_val):

    model_size = len(y_out)
    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()
    batch_sp = []
    for i in range(len(true_arr)): # calculating score for each graph as ranks between 0-3 for every graph
        sp,_ = spearmanr(predict_arr[i,:],true_arr[i,:],nan_policy = 'omit')
        if np.isnan(sp):
            sp = np.nan_to_num(sp)
        # print(kt)
        batch_sp.append(sp)
    return batch_sp
