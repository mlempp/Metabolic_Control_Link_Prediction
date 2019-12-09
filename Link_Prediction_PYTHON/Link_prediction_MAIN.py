# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:30:44 2019

@author: MLempp
"""

import torch
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv, ChebConv, SGConv, RGCNConv, TopKPooling, global_max_pool
from Link_prediction_Utils import Net_Conv, Net_Lin, negative_sampling, val_miss_class, Get_Balanced_Accuracy, evaluate_results, decision_plot, unique_analyse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj
from datetime import date
from datetime import datetime as timer

seed_num = 0
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.backends.cudnn.deterministic= True
torch.cuda.empty_cache()

# =============================================================================
# load data
# =============================================================================
edges = np.array(pd.read_excel (r'graph_structure.xlsx','Edges'))
uniques = np.array(pd.read_excel (r'graph_structure.xlsx','Unique_regs'))
features = np.array(pd.read_excel (r'graph_structure.xlsx','Features_structure')) #NewSS

cross_val_sets      = 5
negative_rate       = 0.5
lr                  = 0.0001
weight_decay        = 1e-3
epochs              = 4000
dropout             = 0.5
N_samples           = 10
kwarg               = MessagePassing(flow = 'source_to_target', aggr = 'add')
device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONV_or_LIN         = 'conv'
gaps                = 'no'
gap_prop            = 0.60
only_mets           = 'no'


TrainResults        = {'recall':[], 'TNR': [], 'FNR':[], 'FPR': [], 'ROC_AUC': [], 'Precision':[],'F1_Score': [], 'Predictions': [], 'Real_class':[], 'Edge':[]}
ValResults          = {'recall':[], 'TNR': [], 'FNR':[], 'FPR': [], 'ROC_AUC': [], 'Precision':[],'F1_Score': [], 'Predictions': [], 'Real_class':[], 'Edge':[]}
Classifications     = {'TP':[], 'FP': [], 'FN':[]}
TestResults        = {'recall':[], 'TNR': [], 'FNR':[], 'FPR': [], 'ROC_AUC': [], 'Precision':[],'F1_Score': [], 'Predictions': [], 'Real_class':[], 'Edge':[]}


if gaps == 'yes':
    ind = np.linspace(2,features.shape[0]-1, features.shape[0]-2).astype(int)
    random.shuffle(ind)
    for i in ind[2:int(len(ind)*gap_prop)]:
        features[i,:] = 0

if only_mets == 'yes':  
    features[314:,2:] = 0



# =============================================================================
# negative sampling and split 
# =============================================================================

edges_and_negatives = negative_sampling(edges, negative_rate = negative_rate)       # negative sampling (n * number of original edges), add an additional row which marks the original network
classes             = edges_and_negatives[2,:] == 1                                 # get logicals for regulation and no regulation
regus               = np.where(classes)[0]                                          # gets regulations
negs                = np.where(~classes)[0]                                         # gets "no" regulation / negatives

random.shuffle(regus)                                                               #shuffle
random.shuffle(negs)
regu_set    = np.array_split(regus, cross_val_sets)                                 # split them in the cross validation sets
negs_set    = np.array_split(negs,  cross_val_sets)

regu_testset = regu_set[cross_val_sets-1].copy()                                    #take last set for testing
negs_testset = negs_set[cross_val_sets-1].copy()

regu_trainvalset = np.concatenate(regu_set[0:cross_val_sets-1].copy())              # merge the remaining
negs_trainvalset = np.concatenate(negs_set[0:cross_val_sets-1].copy())


models = {}
    
# =============================================================================
# take the cross_val_sets-1 and make the training and validation N times
# =============================================================================

for k in range(N_samples):
    print('Run number: ' + str(k+1))                          #do train and validation multiple times and save the models
    models[str(k)] =[]
    
    random.shuffle(regu_trainvalset)                                                        # shuffle
    random.shuffle(negs_trainvalset)
    
    
    for cs in range(cross_val_sets-1):                      # take all sets beside of the last one
    
        print('Cross validation set ' + str(cs+1) + ' of ' + str(cross_val_sets-1))
        
        regu_trainvalset1    = np.array_split(regu_trainvalset.copy(), cross_val_sets-1)               # split them in the cross validation sets
        negs_trainvalset1    = np.array_split(negs_trainvalset.copy(),  cross_val_sets-1)
        
        val_regs    = regu_trainvalset1.pop(cs)
        val_negs    = negs_trainvalset1.pop(cs)
        train_regs  = np.concatenate(regu_trainvalset1)
        train_negs  = np.concatenate(negs_trainvalset1)   
        val_edges   = edges_and_negatives[:, np.concatenate([val_regs, val_negs])]
        train_edges = edges_and_negatives[:, np.concatenate([train_regs, train_negs])]
        
        #setup backbone of network + specific regulations of the train set
        orig_ids            = ((edges_and_negatives[2,:] + edges_and_negatives[3,:]) == 1)
        orig_backbone       = edges_and_negatives[:,orig_ids]
        train_regu          = edges_and_negatives[:,train_regs]       
        train_edges_orig    = np.concatenate((orig_backbone,train_regu),1)                
        
        train_weight    = torch.tensor([1,train_edges.shape[1]/train_edges[2].sum()], dtype = torch.float)
        
        train_data      = torch.tensor(train_edges, dtype = torch.int64)
        val_data        = torch.tensor(val_edges, dtype = torch.int64)
    
    
        #train data
        edge_list   = train_data.clone().type(torch.long)
        x           = np.array(features)
        x           = torch.tensor(x, dtype = torch.float)
        edge_index  = edge_list[0:2,:]
        y           = edge_list[2,:]
        y2          = edge_list[3,:]
        train_edges_orig = torch.tensor(train_edges_orig, dtype = torch.int64)
        data        = Data(x = x, edge_index = edge_index, edge_index_orig = train_edges_orig[0:2,:], y=y, y2 = y2)
        
        #val data
        edge_list   = val_data.clone().type(torch.long)
        x           = np.matrix(features)
        x           = torch.tensor(x, dtype = torch.float) 
        edge_index  = edge_list[0:2,:]
        y           = edge_list[2,:]
        y2          = edge_list[3,:]
        data_val    = Data(x = x, edge_index = edge_index,edge_index_orig = train_edges_orig[0:2,:], y = y, y2 = y2)
        
        
        #submit model to cuda and some hyperparameters
        data            = data.to(device)
        train_weight    = train_weight.to(device)
        data_val        = data_val.to(device)
        
        if CONV_or_LIN == 'conv':
            model           = Net_Conv(kwarg, p=dropout, input_size = data.num_features, output_size = 2, ).to(device)
        elif CONV_or_LIN == 'lin':
            model           = Net_Lin(kwarg, p=dropout, input_size = data.num_features, output_size = 2, ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(),lr = lr, weight_decay = weight_decay)
        
        TrainAccLoss = {'Loss':[], 'Acc': [], 'Crit':[], 'Prec': [], 'F1':[], 'recall':[],'fr':[]}
        ValAccLoss   = {'Loss':[], 'Acc': [], 'Crit':[], 'Prec': [], 'F1':[], 'recall':[],'fr':[]}
        
        plt.figure()
        for epoch in range(1, epochs+1):

           #train
           model.train()
           optimizer.zero_grad()
           out,x2 = model(data, 'Training')
           loss = F.cross_entropy(out, data.y,train_weight)
           TrainAccLoss['Loss'].append(loss)
           loss.backward()
           optimizer.step()
           _, pred = out.max(dim = 1)
           if epoch % 250 == 0:
               print(epoch)

               plt.title('cross validations set: ' + str(cs+1))
               pass
           
           fraction,crit,prec,F1,rec = Get_Balanced_Accuracy(data.y.cpu().view(-1), pred.cpu().view(-1))
           TrainAccLoss['Acc'].append(fraction)
           TrainAccLoss['Crit'].append(crit)
           TrainAccLoss['Prec'].append(prec)
           TrainAccLoss['F1'].append(F1)
           TrainAccLoss['recall'].append(rec)
       
           #validation
           model.eval()
           out1,x2 = model(data_val, 'Validation')
           val_loss = F.cross_entropy(out1, data_val.y).detach()
           ValAccLoss['Loss'].append(val_loss.cpu().item())
           _, pred1 = out1.max(dim = 1)
           fraction1,crit1,prec1,F1,rec = Get_Balanced_Accuracy(data_val.y.cpu().view(-1), pred1.cpu().view(-1))
           
           if epoch % 500 == 0:
               decision_plot(data_val, x2, epoch, epochs)
           ValAccLoss['Acc'].append(fraction1)
           ValAccLoss['Crit'].append(crit1)
           ValAccLoss['Prec'].append(prec1)
           ValAccLoss['F1'].append(F1) 
           ValAccLoss['recall'].append(rec)
        
        recall, TNR, FNR, FPR, ROC_AUC,precision, f1 = evaluate_results(data.y.cpu().view(-1), pred.cpu().view(-1), 1)
        TrainResults['recall'].append(recall)
        TrainResults['TNR'].append(TNR)
        TrainResults['FNR'].append(FNR)
        TrainResults['FPR'].append(FPR)
        TrainResults['ROC_AUC'].append(ROC_AUC)
        TrainResults['Precision'].append(precision)
        TrainResults['F1_Score'].append(f1)
        TrainResults['Predictions'].append(pred)
        TrainResults['Edge'].append(data.edge_index)
        TrainResults['Real_class'].append(data.y)
        
        recall, TNR, FNR, FPR, ROC_AUC,precision, f1 = evaluate_results(data_val.y.cpu().view(-1), pred1.cpu().view(-1), 0)
        find_uniques = unique_analyse(uniques, pred1, data_val)
        ValResults['recall'].append(recall)
        ValResults['TNR'].append(TNR)
        ValResults['FNR'].append(FNR)
        ValResults['FPR'].append(FPR)
        ValResults['ROC_AUC'].append(ROC_AUC)
        ValResults['Precision'].append(precision)
        ValResults['F1_Score'].append(f1)
        ValResults['Predictions'].append(pred1)
        ValResults['Edge'].append(data_val.edge_index)
        ValResults['Real_class'].append(data_val.y)
        
        models[str(k)].append(model)
    
    
        plt.figure()
        ax1=plt.subplot(1, 2, 1)
        ax1.plot(TrainAccLoss['Loss'], '-r',label = 'train_loss')
        ax1.plot(np.array(ValAccLoss['Loss']), '-k', label = 'val_loss')   
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epochs')
        ax1.legend()
        ax1.set_title('cross validations set: ' + str(cs+1))
        
        ax2=plt.subplot(1, 2, 2)
        ax2.plot(np.array(TrainAccLoss['Acc']), '-r', label = 'train_acc')
        ax2.plot(np.array(ValAccLoss['Acc']), '-k',label = 'val_acc' )
        ax2.set_ylabel('acc')
        ax2.set_xlabel('epochs')
        ax2.legend()
        ax2.set_title('uniques: ' + find_uniques)
        
        del model
        

today   = date.today()
d3      = today.strftime("%y%m%d_")
d2      = timer.now().strftime('%H%M%S')
path    = 'results/'
saving  = path+d3+d2+'_epochs_'+str(epochs)+'_'+CONV_or_LIN
np.save(saving, models)


# =============================================================================
# use the  models from different runs and validations sets on the test set
# =============================================================================


train_regs  = regu_trainvalset
test_edges  = edges_and_negatives[:, np.concatenate([regu_testset, negs_testset])]

#setup backbone of network + specific regulations of the train set
orig_ids                = ((edges_and_negatives[2,:] + edges_and_negatives[3,:]) == 1)
orig_backbone           = edges_and_negatives[:,orig_ids]
test_regu               = edges_and_negatives[:,train_regs]       
test_edges_orig         = np.concatenate((orig_backbone,test_regu),1)                

test_data        = torch.tensor(test_edges[:3,:], dtype = torch.int64)

#val data
edge_list   = test_data.clone().type(torch.long)
x           = np.matrix(features)
x           = torch.tensor(x, dtype = torch.float) 
train_edges_orig = torch.tensor(test_edges_orig, dtype = torch.int64)
edge_index  = edge_list[0:2,:]
y           = edge_list[2,:]
data_test   = Data(x = x, edge_index = edge_index,edge_index_orig = train_edges_orig[0:2,:], y = y)
#submit model to cuda and some hyperparameters
data_test        = data_test.to(device)
print('---------------------------------------')
for k in range(N_samples):
    
    m = models[str(k)]

    test_result = []
    for i, model in enumerate(m):
        out1 = model(data_test, 'Validation')[0].detach()
        _, pred1 = out1.max(dim = 1)
        test_result.append(np.array(pred1))
        
    test_result2 = []
    for i,x in enumerate(test_result[0]):
        tmp =[]
        for j in test_result:
            tmp.append(j[i])
        test_result2.append(np.mean(tmp))
    
    pred = [1 if x > 0.5 else 0 for x in test_result2 ]
    find_uniques = unique_analyse(uniques, np.array(pred), data_test)

    recall, TNR, FNR, FPR, ROC_AUC,precision, f1 = evaluate_results(data_test.y.cpu().view(-1), np.array(pred), 2)
    print('uniques: ' + find_uniques)
    TestResults['recall'].append(recall)
    TestResults['TNR'].append(TNR)
    TestResults['FNR'].append(FNR)
    TestResults['FPR'].append(FPR)
    TestResults['ROC_AUC'].append(ROC_AUC)
    TestResults['Precision'].append(precision)
    TestResults['F1_Score'].append(f1)
    TestResults['Predictions'].append(np.array(pred))
    TestResults['Edge'].append(data_test.edge_index)
    TestResults['Real_class'].append(data_test.y)
print('---------------------------------------')

print('N_Samples: {:.4f}'.format(N_samples))
print('VALIDATION')
print('VAL_FNR: {:.4f}'.format(np.mean(ValResults['FNR'])),'std-dev:',np.std(ValResults['FNR']))
print('VAL_FPR: {:.4f}'.format(np.mean(ValResults['FPR'])),'std-dev:',np.std(ValResults['FPR']))
print('VAL_AUC: {:.4f}'.format(np.mean(ValResults['ROC_AUC'])),'std-dev:',np.std(ValResults['ROC_AUC']))
print('VAL_TNR: {:.4f}'.format(np.mean(ValResults['TNR'])),'std-dev:',np.std(ValResults['TNR']))
print('VAL_recall: {:.4f}'.format(np.mean(ValResults['recall'])),'std-dev:',np.std(ValResults['recall']))
print('VAL_precision: {:.4f}'.format(np.mean(ValResults['Precision'])),'std-dev:',np.std(ValResults['Precision']))
print('f1-score: {:.4f}'.format(np.mean(ValResults['F1_Score'])),'std-dev:',np.std(ValResults['F1_Score']))

print('TRAINING')
print('Train_FNR: {:.4f}'.format(np.mean(TrainResults['FNR'])),'std-dev:',np.std(TrainResults['FNR']))
print('Train_FPR: {:.4f}'.format(np.mean(TrainResults['FPR'])),'std-dev:',np.std(TrainResults['FPR']))
print('Train_AUC: {:.4f}'.format(np.mean(TrainResults['ROC_AUC'])),'std-dev:',np.std(TrainResults['ROC_AUC']))
print('Train_TNR: {:.4f}'.format(np.mean(TrainResults['TNR'])),'std-dev:',np.std(TrainResults['TNR']))
print('Train_recall: {:.4f}'.format(np.mean(TrainResults['recall'])),'std-dev:',np.std(TrainResults['recall']))
print('Train_precision: {:.4f}'.format(np.mean(TrainResults['Precision'])),'std-dev:',np.std(TrainResults['Precision']))
print('f1-score: {:.4f}'.format(np.mean(TrainResults['F1_Score'])),'std-dev:',np.std(TrainResults['F1_Score']))

print('TESTING')
print('Test_FNR: {:.4f}'.format(np.mean(TestResults['FNR'])),'std-dev:',np.std(TrainResults['FNR']))
print('Test_FPR: {:.4f}'.format(np.mean(TestResults['FPR'])),'std-dev:',np.std(TrainResults['FPR']))
print('Test_AUC: {:.4f}'.format(np.mean(TestResults['ROC_AUC'])),'std-dev:',np.std(TrainResults['ROC_AUC']))
print('Test_TNR: {:.4f}'.format(np.mean(TestResults['TNR'])),'std-dev:',np.std(TrainResults['TNR']))
print('Test_recall: {:.4f}'.format(np.mean(TestResults['recall'])),'std-dev:',np.std(TrainResults['recall']))
print('Test_precision: {:.4f}'.format(np.mean(TestResults['Precision'])),'std-dev:',np.std(TrainResults['Precision']))
print('f1-score: {:.4f}'.format(np.mean(TestResults['F1_Score'])),'std-dev:',np.std(TrainResults['F1_Score']))

