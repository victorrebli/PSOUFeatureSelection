# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:12:08 2019

@author: reblivi
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings("ignore")
import math
import itertools


class SolutionEvaluatorException(Exception):
                                                 
      def __init__(self,message):
         super(SolutionEvaluatorException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message)   
   
class SolutionEvaluator(object):
    
    
    def __init__(self, p, size_pop):
        
        self.p = p
        self.cv = p.cv
        self.calls = 0
        self.evolution = []
        self.best = 0.0
        self.size_pop = size_pop
        self.n_cols = self.p.n_cols - 1
        self.totaliter = []
        self.data = self.p.data
        self.label = self.p.label
        self.cvs = None
        self.cv = self.p.cv
        self.estimator = self.p.estimator
        self.extra = self.p.extra
        
    def evaluate(self, pop):
        #breakpoint()
        tam = pop['pop'].shape[1] - 1
        x_train = self.data
        y_train = self.label
        dic = dict()
        #breakpoint()
        if y_train is None:
            self.cvs = self._get_cv(np.array(y_train))
        else:
            self.cvs = self._get_cv(y_train)

        for i in range(0, self.size_pop):
            
            lista = self.drop_features(pop['pop'][i, :])
            if(self.p.n_cols - len(lista)) > 1:
                
                #if isinstance(x_train, np.ndarray):
                #    dados_train = np.delete(x_train, np.s_[lista], axis=1)   

                dados_train = x_train.drop(x_train.columns[[lista]], 1)
                    
                pred = self.model(dados_train, y_train)    
                pop['pop'][i ,tam] = pred['mean_ff']
                self.totaliter.append(pop['pop'][i, :])
                if 'mean_cp' in pred.keys():
                    pop['cp'][i] = pred['mean_cp']
                if 'mean_fm' in pred.keys():
                    pop['fm'][i] = pred['mean_fm']

        dic['pop'] = pop['pop']
        if 'mean_cp' in pred.keys():
            dic['cp'] = pop['cp']
        if 'mean_fm' in pred.keys():
            dic['fm'] = pop['fm']


        #breakpoint()
        return dic


    def drop_features(self, s, threshold=0.6):
        
        listt = []
        for i in range(0, len(s) - 1):
            if s[i] <= threshold:
                listt.append(i)

        return listt
    
    
    def model(self, data, y_train):
        
        module_name = self.estimator.__class__
        if 'KMeans' in str(module_name):
            result = self._kmeans(data, y_train)
        else:
            SolutionEvaluatorException(f'Clustering method not implement')
            
        return result


    def _kmeans(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        if self.label is None:

            for train_index, test_index in self.cvs.split(x_tr):
                train_x, valid_x = x_tr[train_index], x_tr[test_index]
                #train_y, valid_y = y_tr[train_index], y_tr[test_index]
                self.estimator.fit(train_x)
                   
                pred_train = self.estimator.predict(train_x)
                pred_test = self.estimator.predict(valid_x)
                lista_models.append(self.metric_intra_inter_cluster(valid_x, pred_test))
            
            return {'mean_ff': np.mean(lista_models)}
        else:
            lista_fm = []
            lista_cp = []
            y_tr = np.array(y_tr)
            
            for train_index, test_index in self.cvs.split(y_tr):    

                train_x, valid_x = x_tr[train_index], x_tr[test_index]
                train_y, valid_y = y_tr[train_index], y_tr[test_index]
                
                self.estimator.fit(train_x)
                pred_train = self.estimator.predict(train_x)
                pred_test = self.estimator.predict(valid_x)

                lista_models.append(self.metric_intra_inter_cluster(valid_x, pred_test))

                lista_cp.append(self.class_purity(valid_y,pred_test))
                lista_fm.append(self.FMeasure(valid_y,pred_test))
            return {'mean_ff': np.mean(lista_models),  'mean_fm': np.mean(lista_fm), 'mean_cp': np.mean([y for x in lista_cp for y in x])}

                         
    def metric_intra_inter_cluster(self, data, pred):

       soma_train_aux_1 = 0
       soma_train_aux_2 = 0
       ####### JW
       for label in np.unique(pred): 
           sub_data = data[np.where(pred == label)]
           media = np.mean(sub_data, axis=0)
           soma_train_aux_1 += (np.power(euclidean_distances(sub_data,media.reshape(1,-1)), 2).sum())

       soma_train_aux_1 = soma_train_aux_1 / len(data)    

       ######## JB
       media_all = np.mean(data,axis=0)
       for label in np.unique(pred):
           sub_data = data[np.where(pred == label)]
           media = np.mean(sub_data,axis=0)
           interm = len(sub_data) / float(len(data)) * np.power(euclidean_distances(media.reshape(1,-1),media_all.reshape(1,-1)), 2)[0][0]
           soma_train_aux_2 += interm

       #### CP
       try:
           perfo_clus = soma_train_aux_2 / float(soma_train_aux_1)
       except:
           raise SolutionEvaluatorException(f'Problem with variables -- soma_train_aux_2:{soma_train_aux_2} - \
                                             soma_train_aux_1:{soma_train_aux_1}')         
       try:
           funfit = math.pow(np.power(self.n_cols,2) - np.power(data.shape[1],2), 1/2.0) / float(self.n_cols)
           funfit = 1 / (1 + np.exp(-funfit))
       except:
           raise  SolutionEvaluatorException(f'Problem with variables -- self.n_cols:{self.n_cols} - \
                                             data.shape::{data.shape[1]}')   
       return (perfo_clus * funfit)

    def class_purity(self, true_labels, pred_labels):
        unicos = np.unique(pred_labels)
        fracoes = []

        for i in range(0, len(unicos)):
            try:
                ind, = np.where(pred_labels == unicos[i])
            except:
                raise SolutionEvaluatorException(f'Problem with variables -- ind:{ind} - unicos:{unicos}')
            try:
                clus = true_labels[ind]
                majority = pd.value_counts(clus).keys()[0]
            except:
                raise SolutionEvaluatorException(f'Problem with variables -- clus:{clus}')

            clus1, = np.where(true_labels == majority)
            clus1 = set(clus)
            ind = set(ind)
            correto = len(ind & clus1)
            fracao = correto / float(len(clus1))
            fracoes.append(fracao)
        return fracoes                    
            
    def FMeasure(self, true_labels, pred_labels):
        
        if len(true_labels) <= 10000:
                ran = range(0, len(pred_labels))
                
                tt = itertools.combinations(ran,2)
                TP, FN, TN, FP = 0, 0, 0, 0
                for idx, i in enumerate(tt):

                    clus1 = pred_labels[i[0]]
                    clus2 = pred_labels[i[1]]
                    clas1 = true_labels[i[0]]
                    clas2 = true_labels[i[1]]
                    if clas1 == clas2:
                        if clus1 == clus2:
                            TP = TP + 1
                        else:
                            FN = FN + 1
                    elif clas1 != clas2:
                        if clus1 == clus2:
                            FP = FP + 1
                        else:
                            TN = TN + 1
                precision = TP / float(TP + FP)
                recal = TP / float(TP + FN)
                FMeasure = 2 * (precision * recal) / float((precision + recal)) 
                return FMeasure 
        else:
            qtd = 100
            list_fm = []
            for _trial in range(0,qtd):
                #print(_trial)
                numbers = np.random.choice(range(0, len(pred_labels)), 1000, replace=False)
                tt = itertools.combinations(numbers,2)
                TP, FN, TN, FP = 0, 0, 0, 0
                for i in tt:
                    clus1 = pred_labels[i[0]]
                    clus2 = pred_labels[i[1]]
                    clas1 = true_labels[i[0]]
                    clas2 = true_labels[i[1]]
                    if clas1 == clas2:
                        if clus1 == clus2:
                            TP = TP + 1
                        else:
                            FN = FN + 1
                    elif clas1 != clas2:
                        if clus1 == clus2:
                            FP = FP + 1
                        else:
                            TN = TN + 1
                precision = TP / float(TP + FP)
                recal = TP / float(TP + FN)
                FMeasure = 2 * (precision * recal) / float((precision + recal))        
                list_fm.append(FMeasure)

            return np.mean(list_fm)    
    
    def _get_cv(self, y_tr):
        
        cv = check_cv(self.cv, y_tr, classifier=is_classifier(self.estimator))
        return cv
                
                    
        
        