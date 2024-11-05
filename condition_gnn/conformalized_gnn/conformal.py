
import numpy as np
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import CoraFull
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.nn import SAGEConv, BatchNorm
import arc
from arc import models
from arc import methods
from arc import black_boxes
from arc import others
from arc import coverage
import torch.optim as optim
import numpy as np
import os.path as osp
import pandas as pd
import numpy as np
import collections
import random
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy.ma.core import maximum
from conditionalconformal.synthetic_data import generate_cqr_data, indicator_matrix
from conditionalconformal import CondConf

from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv


def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    prediction_sets = val_smx >= (1-qhat)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
    
def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    #print(cal_smx[0:20])
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
        
def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False 
    rand = True
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
    
def cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    coverage_flag = (val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])
    cov = coverage_flag.mean()
    """
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    """
    eff = np.mean(val_upper + qhat - (val_lower - qhat))
    return prediction_sets, cov, eff,coverage_flag

def qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    prediction_sets = [val_lower, val_upper]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper - val_lower)
    return prediction_sets, cov, eff

def threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    
    prediction_sets = np.take_along_axis(val_srt <= 1-alpha, val_pi.argsort(axis=1), axis=1)
    prediction_sets[np.arange(prediction_sets.shape[0]), val_pi[:, 0]] = True

    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
#Change the return prediction sets to be false
def run_conformal_classification(pred, data, n, alpha, score = 'aps', 
                                 calib_eval = False, validation_set = False, 
                                 use_additional_calib = False, return_prediction_sets = False, calib_fraction = 0.5,identify =1):
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n
        
    logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
    
    if validation_set:
        smx = logits[data.valid_mask]
        labels = data.y[data.valid_mask].detach().cpu().numpy()
        current_feature =data.x[data.valid_mask].detach().cpu().numpy()
        n_base = int(len(np.where(data.valid_mask)[0])/2)
    else:
        if calib_eval:
            smx = logits[data.calib_test_real_mask]
            current_feature =data.x[data.calib_test_real_mask].detach().cpu().numpy()
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy()
        else:
            smx = logits[data.calib_test_mask]
            current_feature=data.x[data.calib_test_mask].detach().cpu().numpy()
            labels = data.y[data.calib_test_mask].detach().cpu().numpy()

    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    #store each idx for every k and store the single smx and labels do the inference after. rememver only store the information of idx if we see identify 2 or 3.
    for k in range(100):
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        #print(cal_smx.shape)
        #print(cal_smx[0:10])
        cal_labels, val_labels = labels[idx], labels[~idx]
        val_feature =current_feature[~idx]
        if use_additional_calib and calib_eval:
            smx_add = logits[data.calib_eval_mask]
            labels_add = data.y[data.calib_eval_mask].detach().cpu().numpy()
            cal_smx = np.concatenate((cal_smx, smx_add))
            cal_labels = np.concatenate((cal_labels, labels_add))
            
        n = cal_smx.shape[0]
        
        if score == 'tps':
            prediction_sets, cov, eff = tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'aps':
                prediction_sets, cov, eff = aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
                #import pickle
                #cal_features = current_feature[idx]
                #test_features = current_feature[~idx]
                #test_labels = val_labels
                #print(identify)
                #if identify==2:
                        #variables_to_save = {'cal_features': cal_features, 'cal_labels': cal_labels,'test_features':test_features,'test_labels':test_labels,'idx':idx,'num_features':data.num_features,'prediction':smx,'num_classes':smx.shape[1]}
                       # serialized_file_name = 'run_condition_variables_CFGNN_2.pkl'
                       # serialized_file_path = './' + serialized_file_name  # This specifies the current directory

                       # with open(serialized_file_path, 'wb') as file:
                            #pickle.dump(variables_to_save, file)
                        
               # elif identify ==3:
                       # variables_to_save = {'cal_features': cal_features, 'cal_labels': cal_labels,'test_features':test_features,'test_labels':test_labels,'idx':idx,'num_features':data.num_features,'prediction':smx,'num_classes':smx.shape[1]}
                       # serialized_file_name = 'run_condition_variables_GNN.pkl'
                       # serialized_file_path = './' + serialized_file_name  # This specifies the current directory

                       # with open(serialized_file_path, 'wb') as file:
                        #    pickle.dump(variables_to_save, file)

                
            
        elif score == 'raps':
            prediction_sets, cov, eff = raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'threshold':
            prediction_sets, cov, eff = threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
            
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)
    if identify==2:
           print('identify 2')
           variables_to_save = {'softmax': smx, 'index_100':idx_all,'labels':labels}
           serialized_file_name = 'run_condition_variables_identify_2.pkl'
           serialized_file_path = './' + serialized_file_name
           with open(serialized_file_path, 'wb') as file:
                pickle.dump(variables_to_save, file)
           variables_to_save = {'prediction_set':pred_set_all,'labels':labels,'index_100':idx_all,'softmax': smx}
           serialized_file_name = 'cfgnn_results_identify_2.pkl'
           serialized_file_path = './' + serialized_file_name
           with open(serialized_file_path, 'wb') as file:
                pickle.dump(variables_to_save, file)           
    if return_prediction_sets:
        return cov_all, eff_all, prediction_sets, val_labels,val_feature
    else:
        return np.mean(cov_all), np.mean(eff_all)

def run_conformal_regression(pred, data, n, alpha, calib_eval = False, validation_set = False, use_additional_calib = False, return_prediction_sets = True, calib_fraction = 0.5, score = 'cqr'):
    
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n
    
    try:
        pred = pred.detach().cpu().numpy()
    except:
        pass
                     
    if validation_set:
        smx = pred[data.valid_mask]
        labels = data.y[data.valid_mask].detach().cpu().numpy().reshape(-1)
        current_feature =data.x[data.valid_mask].detach().cpu().numpy()
        n_base = int(len(np.where(data.valid_mask)[0])/2)
    else:
        if calib_eval:
            smx = pred[data.calib_test_real_mask]
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy().reshape(-1)
            current_feature =data.x[data.calib_test_real_mask].detach().cpu().numpy()
        else:
            smx = pred[data.calib_test_mask]
            labels = data.y[data.calib_test_mask].detach().cpu().numpy().reshape(-1)
            current_feature=data.x[data.calib_test_mask].detach().cpu().numpy()
    import pickle
    with open('./labelsandfeatures.pkl', 'wb') as f:
        pickle.dump({'labels': labels,'current_feature':current_feature, 'features': data.x.detach().cpu().numpy(),'calib_test_mask':data.calib_test_mask}, f)
    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(1):
        upper, lower = smx[:, 2], smx[:, 1]

        idx = np.array([1] * n_base + [0] * (labels.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_upper, val_upper = upper[idx], upper[~idx]
        cal_lower, val_lower = lower[idx], lower[~idx]
        if score == 'cqr':
            prediction_sets, cov, eff,coverage_flag = cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha)
            #I have 1.pred which is the prediction 2.calibration features and labels 3. test features and labels4. score function
            cal_features = current_feature[idx]
            test_features = current_feature[~idx]
            test_labels = val_labels
            import pickle
            # Serialize your variables
            if calib_eval:
                variables_to_save = {'cal_features': cal_features, 'cal_labels': cal_labels,'test_features':test_features,'test_labels':test_labels,'idx':idx,'num_features':data.num_features,'prediction':smx}
                serialized_file_name = 'run_condition_variables_CFGNN_3.pkl'
                serialized_file_path = './' + serialized_file_name

                with open(serialized_file_path, 'wb') as file:
                    pickle.dump(variables_to_save, file)

            else:
                variables_to_save = {'cal_features': cal_features, 'cal_labels': cal_labels,'test_features':test_features,'test_labels':test_labels,'idx':idx,'num_features':data.num_features,'prediction':smx}
                serialized_file_name = 'run_condition_variables_GNN.pkl'
                
                serialized_file_path = './' + serialized_file_name

                with open(serialized_file_path, 'wb') as file:
                    pickle.dump(variables_to_save, file)

                
        elif score == 'qr':
            prediction_sets, cov, eff = qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha)
            
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)


    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels, idx_all,coverage_flag,current_feature[~idx]
    else:
        return np.mean(cov_all), np.mean(eff_all)
    
    
    
