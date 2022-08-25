

'''Import statements'''
import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
import math
from sklearn.utils import shuffle

import os
import sys
from utils import *
from trainnupack import *
from oracle import * 





'''This script includes the neural network models

> Inputs: list of DNA sequences in letter format '''


class modelNet():
    '''Runs the Proxy Model with Nupack data'''
    def __init__(self, config, ensembleIndex):
        
        self.config = config
        self.ensembleIndex = ensembleIndex
        self.config.history = min(config.proxy_history, self.config.proxy_max_epochs) # length of past to check
        self.config.training_batch_size = self.config.proxy_training_batch_size

        self.initproxyModel()
        torch.random.manual_seed(int(config.model_seed + ensembleIndex))

        self.scheduler1 = lr_scheduler.ReduceLROnPlateau(  #reduce learning rate once metric stopping improving
            self.optimizer,
            mode='min', #once stopped decreasing 
            factor=0.5, #why this?, lr reduction factor 
            patience=10, #epocs to ignore with no improvement, set to default - Micheal's was different
            threshold = 1e-3,
            threshold_mode = 'rel',
            cooldown=0  #self.config.history // 2, set to default
        )

    # def training_batch_size(self): #include micheal's implementation of trying different batch sizes
    #     finished=0
    #     training_batch_0 = 1 * self.config.proxy_training_batch_size
        
    #     final_batch_size =int(self.config.proxy_training_batch_size) # Michael had a different way of getting this

    #     print('Final batch size is {}'.format(final_batch_size))

    def initproxyModel(self):
        '''
        Initialize proxy model and optimizer
            '''
        if self.config.proxy_model_type=='mlp':
            self.model=MLP(self.config)
        else:
            print(self.config.proxy_model_type+'is not an available model')
        
        if self.config.device=='cuda':
            self.model=self.model.cuda() 
        self.optimizer=optim.AdamW(self.model.parameters(),amsgrad=True)
    
                
    def converge(self,dataset,returnHist=False):
        '''train until test loss converges'''
        #no batch sizing to include 

        self.initproxyModel() #reset model        
        [self.err_tr_hist, self.err_te_hist] = [[], []] # initialize error records
        
        tr, te, self.datasetSize = getDataLoaders(self.config, self.ensembleIndex,dataset)
     
        self.converged = 0 # convergence flag
        self.epochs = 0
        
        while (self.converged != 1): 
            t0=time.time()
            if self.epochs > 0: #  this allows us to keep the previous model if it is better than any produced on this run
                self.train_net(tr)
            else:
                self.err_tr_hist.append(torch.zeros(1).to(self.config.device)[0])
              #  self.tar_tr_hist.append(torch.zeros(1).to(self.config.device)[0])
            self.test(te) #why??
            tf=time.time()
            # after training at least 10 epochs, check convergence
            if self.epochs >= self.config.history:
                self.checkConvergence()
            if True:#(self.epochs % 10 == 0):
                print("Model {} epoch {} train loss {:.3f} test loss {:.3f} took {} seconds".format(self.ensembleIndex, self.epochs, self.err_tr_hist[-1], self.err_te_hist[-1], int(tf-t0)))
            self.epochs += 1
        if returnHist:
            return torch.stack(self.err_tr_hist).cpu().detach().numpy(), torch.stack(self.err_te_hist).cpu().detach().numpy()

    def checkConvergence(self):
        '''check that the model converged'''
        eps = 1e-4 # relative measure for constancy

        if all(torch.stack(self.err_te_hist[-self.config.history+1:])  > self.err_te_hist[-self.config.history]): #
            self.converged = 1
            print(bcolors.WARNING + "Model converged after {} epochs - test loss increasing at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)


        # check if train loss is unchanging
        lr0 = np.copy(self.optimizer.param_groups[0]['lr'])
        self.scheduler1.step(torch.mean(torch.stack(self.err_tr_hist[1:])))  # plateau scheduler, skip first epoch
        lr1 = np.copy(self.optimizer.param_groups[0]['lr'])
        if lr1 != lr0:
            print('Learning rate reduced on plateau from {} to {}'.format(lr0, lr1))

        if abs(self.err_tr_hist[-self.config.history] - torch.mean(torch.stack(self.err_tr_hist[-self.config.history:])))/self.err_tr_hist[-self.config.history] < eps:
            self.converged = 1
            print(bcolors.WARNING + "Model converged after {} epochs - hit train loss convergence criterion at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        # check if we run out of epochs
        if self.epochs >= self.config.proxy_max_epochs:
            self.converged = 1
            print(bcolors.WARNING + "Model converged after {} epochs- epoch limit was hit with test loss {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)
    
        
    def train_net(self,tr):
        '''perform one epoc of training '''
        err_tr=[]
        tr_values=[]
        self.model.train(True)
        for i, trainData in enumerate(tr):
            proxy_loss=self.getLoss(trainData)
            err_tr.append(proxy_loss.data)
            self.optimizer.zero_grad()  #set gradients to zer0 #where do these gradient calcs go? 
            proxy_loss.backward()
            self.optimizer.step()
            

        self.err_tr_hist.append(torch.mean(torch.stack(err_tr)))
  
    def test(self,te):
        ''' get the loss over the test set '''
        err_te=[]
        self.model.train(False)
        with torch.no_grad():
            for i,testData in enumerate(te):
                loss=self.getLoss(testData)
                err_te.append(loss)
                
        self.err_te_hist.append(torch.mean(torch.stack(err_te)))
    
            
    def getLoss(self,train_data):
        '''get the regression loss on a batch of datapoints '''
        inputs=train_data[0]
        targets=train_data[1]
        self.inputs=inputs
        self.targets=targets 
        if self.config.device == 'cuda':
                inputs = inputs.cuda()
                targets = targets.cuda()
        
        output = self.model(inputs.float())

        return F.mse_loss(output[:,0], targets.float())


    def raw(self, Data, output="Average"): #todo
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        if self.config.device == 'cuda':
            Data = torch.Tensor(Data).cuda().float()
        else:
            Data = torch.Tensor(Data).float()

        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(Data).cpu().detach().numpy()  #converting tensor to numpy for tranformation 
            if output == 'Average':
                return np.average(out,axis=1)
            elif output == 'Variance':
                return np.var(out,axis=1)
            elif output == 'Both':
                return np.average(out,axis=1), np.var(out,axis=1)


class buildDataset():  #to include datashuffling (but no need right now)
    '''
    Builds train and test dataset by splitting full set 
    '''
    def __init__(self,config,dataset=None):
        if dataset is None:
                dataset = np.load('nupack_dataset.npy',allow_pickle=True).item()
        self.samples = dataset['samples']
        
        self.targets = dataset['scores']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
    def getFullDataset(self):
        return self.samples, self.targets

def getDataLoaders(config, ensembleIndex,dataset): #not sure what ensemble ine is 
        training_batch=config.proxy_training_batch_size
        dataset=buildDataset(config,dataset) #get data 
      #  dataset = datasetbuilder.getFullDataset()
        train_size = int(0.5 * len(dataset))  # split data into training and test sets
        test_size=len(dataset)-train_size
        #construct loaders for inputs and targets 
        train_dataset = []
        test_dataset = []
        
        for i in range(test_size,test_size+train_size): # get training data from dataset end
            train_dataset.append(dataset[i])
            
        for i in range(test_size):
            test_dataset.append(dataset[i]) #i dont include the cutoff value here 
        tr=data.DataLoader(train_dataset,batch_size=5,shuffle=False,num_workers=0,pin_memory=False)
        te = data.DataLoader(test_dataset,batch_size=5, shuffle=False, num_workers= 0, pin_memory=False) #why is batch size same?
             
        return tr, te, dataset.__len__()

class MLP(nn.Module):
    def __init__(self,config):
        super(MLP,self).__init__()
        
        if True:
            act_func='tanh'
        
        self.inputLength=config.max_sample_length
        
        #main MLP params 
        self.layers =config.proxy_model_layers #hidden layers 
        self.filters=config.proxy_model_width
        self.embedding_mode = config.MLP_embedding
        self.classes = int(config.dataset_dict_size + 1)


        #input layers:
        if config.MLP_embedding == 'one hot':
            self.init_layer_width = int(self.inputLength * self.classes)
            self.initial_layer = nn.Linear(self.init_layer_width, self.filters) # layer which takes in our sequence in one-hot encoding

        elif config.MLP_embedding == 'embed':
            self.embedDim = config.proxy_model_embedding_width
            self.dictLen = config.dataset_dict_size
            self.embedding = nn.Embedding(self.dictLen + 1, embedding_dim=self.embedDim)
            self.init_layer_depth = int(self.embedDim * self.inputLength)
            self.initial_layer = nn.Linear(self.init_layer_depth, self.filters) # layer which takes in our sequence in one-hot encoding

        #defining activation function
        self.activation1=Activation(act_func,self.filters,config)
        # output layer definition
        self.output_layers = nn.Linear(self.filters, 1, bias=False)  #maybe change the width of these layers - this could be formatted 

        #hidden layers params
        self.lin_layers=[]
        self.norms=[]
        self.dropouts=[]
        self.activations=[]

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters,self.filters))
            self.activations.append(Activation(act_func,self.filters))
            if config.proxy_norm == 'batch':
                self.norms.append(nn.BatchNorm1d(self.filters))
            elif config.proxy_norm =='layer':
                self.norms.append(nn.LayerNorm(self.filters))

            if config.proxy_dropout_prob != 0:
                self.dropouts.append(nn.Dropout(config.proxy_dropout_prob))
            else:
                self.dropouts.append(nn.Identity())                
        
        # initialize module lists 
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        self.norms = nn.ModuleList(self.norms)
        self.dropouts = nn.ModuleList(self.dropouts)
        
    def forward(self,x,clip=None): 
            if self.embedding_mode=='one hot':
                x=F.one_hot(x.long(),num_classes=self.classes)
                x=x.reshape(x.shape[0],self.init_layer_depth).float()
            elif self.embedding_mode=='embed':
                x=self.embedding(x.long())
                x=x.reshape(x.shape[0],self.init_layer_depth).float()
            
            x = self.activation1(self.initial_layer(x)) # apply linear transformation and nonlinear activation on input
            for i in range(self.layers):
                residue = x.clone()
                x = self.lin_layers[i](x)
                x = self.norms[i](x)
                x = self.dropouts[i](x)
                x = self.activations[i](x)
                x = x + residue
                
            x = self.output_layers(x) # each task has its own head
            if clip is not None:
                x=torch.clip(x,max=clip)

            return x
        
class Activation(nn.Module):
    def __init__(self,activation_func,filters, *args,**kwargs):
        super().__init__()
        if activation_func=='relu':
            self.activation=F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'tanh':
            self.activation = F.tanh
            
    def forward(self,input):
        return self.activation(input)
   