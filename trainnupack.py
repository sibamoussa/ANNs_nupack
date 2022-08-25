from distutils.command.install_scripts import install_scripts
import os 
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts
from matplotlib.pyplot import figlegend 
import torch
from torch import nn, optim, cuda, backends
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.utils import data
import torch.nn.functional as F
import time
from utils import *
import tqdm
from oracle import *
from models import *
'''
to do:
1) test batch sizes 
2) shuffle dataset 
3) incorp mlflow
4) retrun histories 
5) why is cuda sometimes used and sometimes no 
6)         standardized_target_max = torch.amax(targets)
7) i havent figured out the ensemble index
8) evalute func not included - no reasoning behind standadization 
9) remove the mean and the std 
10) include batch size in the train and test dataloaders set 
11) look into a tuner for hyperparams
'''

try: 
    from nupack import *
except: 
    pass 


class Trainer():
    def __init__(self,config):
        self.config=config
        print('Started dataset generation')
        oracle=Oracle(config)
        t0=time.time()
        self.dataset=oracle.initializeDataset(returnData=True)
        print('Building dataset took {} seconds'.format(time.time()-t0))
        datasetBuilder=buildDataset(self.config,self.dataset)
        
    def train(self):
        test_loss = []
        train_loss = []
        for j in range(self.config.proxy_model_ensemble_size):
            print('Training model {}'.format(j))
            self.resetModel(j)
            err_tr_hist, err_te_hist = self.model.converge(self.dataset, returnHist=True)
            train_loss.append(err_tr_hist)
            test_loss.append(err_te_hist)
            return err_tr_hist, err_te_hist

    def resetModel(self,ensembleIndex, returnModel=False):
        '''load a new instance of the model make sure params are reset'''
        try:  #delete previous model 
            del self.model    
        except: 
            self.model = modelNet(self.config,ensembleIndex)
        if returnModel:
            return self.model
          