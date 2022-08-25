
from distutils.command.install_scripts import install_scripts
import os 
from random import random, randint
from matplotlib.pyplot import figlegend 
import torch
from torch import nn, optim, cuda, backends
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.utils import data
import torch.nn.functional as F
import time
from utils import *
from trainnupack import*
from models import *
from oracle import *
from argparse import ArgumentParser
import tqdm
import os
from sklearn.metrics import confusion_matrix


os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
'''

'''
to do: 
1) do I need to import all from the scripts or can i just import one?
2) no params included in the proxy model 
3) use different widths for different layers 
4) look up positional embedding
5) twin net?
6) shuffling of dataset 

'''

try: 
    from nupack import *
except: 
    pass 

def add_bool_arg(parser,name,default=False):  #not sure I understand this part
    group=parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--'+name,dest=name,action='store_true')
    group.add_argument('--no-'+name,dest=name,action='store_false')
    parser.set_defaults(**{name:default})
    
def add_args(parser):

    '''
    
    Adds command-line arguments to parser
    Returns:
        argparse.Namespace: the parsed arguements
        
    '''
    args2config = {}
    # YAML config
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
  
        
    parser.add_argument("--run_num",type=int,default=0,help="Experiment ID")
    parser.add_argument("--device", default="cpu", type=str, help="'cuda' or 'cpu'")


    #oracle to build a new dataset 
    parser.add_argument("--dataset_size",type=int,default=int(2000),help="number of items in the initial (toy) dataset") # NOTE dataset is split 10:90 train:test
    parser.add_argument("--dataset_dict_size",type=int,default=4,help="number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4 - with variable length, 0's are added for padding")

    parser.add_argument("--oracle", type=str, default="nupack energy")  # 'linear' 'potts' 'nupack energy' 'nupack pairs' 'nupack pins'
    parser.add_argument("--min_sample_length", type=int, default=30)
    parser.add_argument("--max_sample_length", type=int, default=30)

    #proxy model
    parser.add_argument("--proxy_model_type",type=str,default="mlp",  help="type of proxy model - mlp or transformer")
    parser.add_argument("--MLP_embedding", type=str, default='embed', help="embed or one hot")
    parser.add_argument("--proxy_model_width",type=int,default=64,help="number of neurons per proxy NN layer")
    parser.add_argument("--proxy_model_layers",type=int,default=4,help="number of layers in NN proxy models (transformer encoder layers OR MLP layers)")
    parser.add_argument("--proxy_training_batch_size", type=int, default=100) #not sure about this
    add_bool_arg(parser, "auto_batch_sizing", default=True)
    parser.add_argument("--proxy_max_epochs", type=int, default=1000) #what is the real default?
    add_bool_arg(parser, 'proxy_shuffle_dataset', default=True)
    add_bool_arg(parser, 'proxy_clip_max', default=False) #not sure about this 
    parser.add_argument("--proxy_dropout_prob", type=float,default=0) #[0,1) dropout probability on fc layers, how is this set?
    parser.add_argument("--proxy_norm", type=str,default='layer') # None, 'batch', 'layer', not sure about this
    parser.add_argument("--proxy_init_lr", type=float, default = 1e-5) #not sure about this
    parser.add_argument("--proxy_history", type=int, default = 100)
    parser.add_argument("--proxy_model_ensemble_size",type=int,default=1,help="number of models in the ensemble")
    parser.add_argument("--proxy_model_embedding_width", type=int, default=64) # depth of transformer embedding


    return parser
#log paramters initially and then worry about everything else 
    
def process_config(config):
    # Normalize seeds
    config.model_seed =0 #config.model_seed % 10
    config.dataset_seed = 0 # = config.dataset_seed % 10
    config.toy_oracle_seed = 0 #config.oracle_seed % 10

    return config


if __name__=="__main__":
    #log a paramter (key-va lue pair)
    parser = ArgumentParser()
    parser = add_args(parser)
    config = parser.parse_args()
    config = process_config(config)
    trainer = Trainer(config)
    err_tr, err_te = trainer.train()

    # visualize predictions
    tr, te, datasetSize = getDataLoaders(trainer.model.config,0, trainer.dataset)
    preds_tr = []
    truth_tr = []
    for i, trainData in enumerate(tr): 
        preds_tr.extend(trainer.model.model(trainData[0]).cpu().detach().numpy())
        truth_tr.extend(trainData[1].cpu().detach().numpy())
    preds = np.asarray(preds_tr)
    truth = np.asarray(truth_tr)
    
    residuals_train=residCalc(preds,truth)

    #test set 
    preds_te = []
    truth_te = []
    for i, testData in enumerate(te):
        preds_te.extend(trainer.model.model(testData[0]).cpu().detach().numpy())
        truth_te.extend(testData[1].cpu().detach().numpy())
    preds_test = np.asarray(preds_te)
    truth_test = np.asarray(truth_te)
    
    residuals_test=residCalc(preds_test,truth_test)

    #model accuracy
    acc_train=calcAccuracy(preds,truth)
    acc_test=calcAccuracy(preds_test,truth_test)
   
    mse_train=mseCalc(preds,truth)
    mse_test=mseCalc(preds_test,truth_test)
    print('msetrain at last epoc: {}'.format(mse_train))
    print('msetest at last epoc: {}'.format(mse_test))

    mae_train=maeCalc(preds,truth)
    mae_test=maeCalc(preds_test,truth_test)
    print('maetrain at last epoc: {}'.format(mae_train))
    print('metest at last epoc: {}'.format(mae_test))
    
    conf_mat=rmse(truth, preds)
    print(conf_mat)

    plt.clf()
    le=3
    wd=3
    plt.subplot(wd,le,4)
    
    plt.subplot(wd,le,1)
    plt.plot(range(0,len(err_tr)),err_tr,label='training loss')
    plt.legend()
    
    plt.subplot(wd,le,2)
    plt.plot(range(0,len(err_te)),err_te,label='test loss')
    plt.legend()
    
    plt.subplot(wd,le,3)
    plt.scatter(preds,truth,label=' accuracy {}'.format(acc_train))
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.legend()
    
    plt.subplot(wd,le,4)
    plt.scatter(preds_test,truth_test,label='accuracy{}'.format(acc_test))
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.legend()

    plt.subplot(wd,le,5)
    plt.scatter(range(0,len(residuals_train)),residuals_train)
    plt.legend()

    plt.subplot(wd,le,6)
    plt.scatter(range(0,len(residuals_test)),residuals_test)

    plt.legend()
    #plt.plot(np.linspace(np.amin(truth), np.amax(truth), 100), np.linspace(np.amin(truth), np.amax(truth), 100), 'k.-')
    plt.show()
    
