# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:30:44 2019

@author: MLempp
"""

import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, ChebConv, SGConv, RGCNConv
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
from numpy import random
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.colors import ListedColormap
from torch_geometric.utils import dropout_adj
from torch_geometric.nn.conv import MessagePassing

kwarg               = MessagePassing(flow = 'source_to_target', aggr = 'add')

class Net_Conv(torch.nn.Module):
    def __init__(self,kwarg, p=0, input_size=1, output_size=2):
        super(Net_Conv,self).__init__()
        self.p          = p
        self.drop1      = torch.nn.Dropout(p = p)
        self.conv1      = GCNConv(input_size,5,kwarg, cached = True)
        #self.conv1      = SAGEConv(input_size,2,normalize=False)
        #self.conv1      = ChebConv(input_size,2,K=2)
        self.linear1    = torch.nn.Linear(int(input_size)*2,4,bias = True)
        self.linear2    = torch.nn.Linear(10,2, bias = True)
        
    def forward(self, data, flag):
        x, edge_index, edge_orig  = data.x, data.edge_index, data.edge_index_orig
        if flag == 'Training':
           edge_index1, _ = dropout_adj(edge_orig,p=self.p)   
           edge_index2, _ = dropout_adj(edge_orig,p=self.p)        
        else:
           edge_index1, _ = dropout_adj(edge_orig,p=0)
           edge_index2, _ = dropout_adj(edge_orig,p=0)

        x = self.conv1(x, edge_index1)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        
        z = x[edge_index[0,:],:]
        y = x[edge_index[1,:],:]
        
#        x = torch.add(z,y)/2
#        x = torch.cat((y,z), 1)    
#        x = F.cosine_similarity(z,y)
#        x = x[:,None]
        
        x = torch.cat((y,z), 1)
        
#        x = self.linear1(x)
#        x = self.drop1(x)
#        x = F.relu(x)
        
        x2 = self.linear2(x)
        x = x2
        return x, x2
    
    
class Net_Lin(torch.nn.Module):
    def __init__(self,kwarg, p=0, input_size=1, output_size=2):
        super(Net_Lin,self).__init__()
        self.p          = p
        self.drop1      = torch.nn.Dropout(p = p)
        self.linear1    = torch.nn.Linear(input_size,5,bias = True)
        self.linear2    = torch.nn.Linear(10,2, bias = True)
        
    def forward(self, data, flag):
        x, edge_index, edge_orig  = data.x, data.edge_index, data.edge_index_orig
        if flag == 'Training':
           edge_index1, _ = dropout_adj(edge_orig,p=self.p)   
           edge_index2, _ = dropout_adj(edge_orig,p=self.p)        
        else:
           edge_index1, _ = dropout_adj(edge_orig,p=0)
           edge_index2, _ = dropout_adj(edge_orig,p=0)

        x = self.linear1(x)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        
        z = x[edge_index[0,:],:]
        y = x[edge_index[1,:],:]
        
#        x = torch.add(z,y)/2
#        x = torch.cat((y,z), 1)    
#        x = F.cosine_similarity(z,y)
#        x = x[:,None]
        
        x = torch.cat((y,z), 1)
        
#        x = self.linear1(x)
#        x = self.drop1(x)
#        x = F.relu(x)
        
        x2 = self.linear2(x)
        x = x2
        return x, x2


def negative_sampling(all_edges, negative_rate):
    
    met_nodes   = np.unique(all_edges[0,:])
    rxn_nodes   = np.unique(all_edges[1,:])
    edge        = np.array([[],[]])
    edge_tuple  = [tuple(x) for x in np.transpose(edge)]
    exclusions  = [tuple(x) for x in np.transpose(np.array(all_edges[0:2,:]))]
    

    while edge.shape[1] < np.round(negative_rate * all_edges.shape[1]):
        rnd_met_id = np.random.randint(len(met_nodes), size = 1)
        rnd_rxn_id = np.random.randint(len(rxn_nodes), size = 1)
        
        source_node = met_nodes[rnd_met_id]
        target_node = rxn_nodes[rnd_rxn_id]
        tentative_edge = tuple((source_node[0],target_node[0]))   
        if tentative_edge not in edge_tuple and exclusions:
           edge_tuple.append(tentative_edge)
           edge = np.array(np.transpose(edge_tuple))
    
    negative_edges = edge
    edge_class = np.zeros((1, negative_edges.shape[1])).astype(int)
    negative_edges = np.concatenate((negative_edges,edge_class), axis = 0)
    
    pos = np.ones((1, all_edges.shape[1])).astype(int)
    neg = np.zeros((1, negative_edges.shape[1])).astype(int)
    
    all_edges       = np.concatenate((all_edges, pos), axis = 0)
    negative_edges  = np.concatenate((negative_edges, neg), axis = 0)
    
    edges = np.concatenate((all_edges,negative_edges), axis = 1)
    return edges


def Get_Balanced_Accuracy(target, prediction): #is the same as AUC
    tn, fp, fn, tp= confusion_matrix(target,prediction).ravel()
    recall = tp/(tp+fn)
    TNR = tn/(tn+fp)
    Balanced_Accuracy = (recall + TNR)/2 #equals AUC
    criterium2 = recall-(1-TNR)
    precision = tp/(tp+fp)
    F1 = 2*precision*recall/(precision+recall)
    return Balanced_Accuracy, criterium2, precision, F1,recall


def val_miss_class(target,prediction,edges):
    prediction = np.array(prediction)
    target = np.array(target)
    edges = np.array(edges)
    pos_pred = prediction == 1
    neg_pred = prediction == 0
    
    pos_pred = np.reshape(pos_pred,(pos_pred.shape[0]))
    
    edges_pos = edges[:,pos_pred]
    edges_neg = edges[:,neg_pred]

    fp_id = target[pos_pred] == 0
    tp_id = target[pos_pred] == 1
    fn_id = target[neg_pred] == 1
    
    fp_edges = edges_pos[:,fp_id]
    fn_edges  = edges_neg[:,fn_id]
    tp_edges = edges_pos[:,tp_id]
    return fp_edges, fn_edges, tp_edges


def decision_plot (data_set, x2, epoch, epochs):
    yks = data_set.y.cpu().numpy()# +  data_set.y2.cpu().numpy()
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    colors= ['cyan', 'red', 'blue'] 
    X=x2.detach().numpy()          
    ax1=plt.subplot(1, epochs/500, (epochs/500)/(epochs/epoch))
    h = .02
    x_min, x_max = -2,3 #X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 
    y_min, y_max = -2,3 #X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    newdata = np.c_[xx.ravel(), yy.ravel()]
    XX = torch.Tensor(newdata)
    yhat = F.log_softmax(XX, dim=1)
    _,hsmf2 = yhat.max(dim=1)
    yhat = hsmf2.numpy().reshape(xx.shape)
    #plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    ax1.plot([-3, 3], [-3,3], 'k-')
    ax1.scatter(X[:,0], X[:,1], c=yks, cmap=ListedColormap(colors), s= 1)
    ax1.set_ylim([-3,3])
    ax1.set_xlim([-1.5,3])


def evaluate_results(target, prediction,caze):
    
    fpr, tpr, _ = roc_curve(target, prediction)
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp= confusion_matrix(target,prediction).ravel()
    recall = tp/(tp+fn)
    TNR = tn/(tn+fp)
    FNR = 1-recall
    FPR = 1-TNR
    Balanced_Accuracy = (recall+TNR)/2
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    if caze == 1:
       print('Train Results')
    elif caze == 2:
        print('Test Results')
    else:
        print('Validation Results')        

    print('Recall/ True Positive Rate: {:.4f}'.format(recall))
    print('Specificity/ True Negative Rate: {:.4f}'.format(TNR))
    print('False Negative Rate: {:.4f}'.format(FNR))
    print('False Positive Rate: {:.4f}'.format(FPR))
    print('ROC_AUC: {:.4f}'.format(Balanced_Accuracy))
    return recall, TNR, FNR, FPR, Balanced_Accuracy, precision, f1

def unique_analyse(uniques, p, data_set):
    
    edge_set    = [tuple(x) for x in np.transpose(np.array(data_set.edge_index))]
    edge_unique = [tuple(x) for x in np.transpose(np.array(uniques))]
    inter = list(set(edge_set) & set(edge_unique))
    ind = [edge_set.index(a) for a in inter]
    unique_pred = p[ind]
    if np.array(unique_pred.sum()) > 0:
        answer = 'yes'
    else:
        answer = 'no'
    return answer