import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from bayes_opt import BayesianOptimization
from prepare_data import *
import torch.nn.functional as F
# from evaluate_func import *

# torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def cal_r2_score(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2) 
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2) 
    r2 = 1 - (ss_res / ss_tot)  
    return r2

def cal_pearson_correlation(y_pred, y_true):
    y_pred_mean = torch.mean(y_pred)
    y_true_mean = torch.mean(y_true)
    numerator = torch.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) 
    denominator = torch.sqrt(torch.sum((y_pred - y_pred_mean) ** 2) * torch.sum((y_true - y_true_mean) ** 2))
    return numerator / denominator 

def cal_mape(y_pred, y_true):
    mask = y_true != 0 
    y_pred_filtered = y_pred[mask]
    y_true_filtered = y_true[mask]
    return torch.mean(torch.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

def cal_mpe(y_pred, y_true):
    mask = y_true != 0 
    y_pred_filtered = y_pred[mask]
    y_true_filtered = y_true[mask]
    return torch.mean((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100


def evaluate(score, label):
    score, label = torch.tensor(score), torch.tensor(label)
    mse = torch.mean(torch.abs(score - label) ** 2)
    mae = torch.mean(torch.abs(score - label))
    rmse = np.sqrt(mse)
    r2, pea, mape, mpe = cal_r2_score(score, label), cal_pearson_correlation(score, label), cal_mape(score, label), cal_mpe(score, label)
    return mae, mse, rmse, r2, pea, mape, mpe
