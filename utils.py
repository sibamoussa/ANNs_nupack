
"""import statement"""
from argparse import Namespace
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sklearn.cluster as cluster


"""
This is a general utilities file for the proxy model pipeline 

"""
class bcolors:  

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
def get_n_params(model):
    """
    count parameters for a pytorch model
    :param model:
    :return:
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def filterDuplicateSamples(samples, oldDatasetPath=None, returnInds=False):
    """
    assumes original dataset contains no duplicates
    :param samples: must be np array padded to equal length. If a combination of new and original datasets, critical that the original data comes first.
    : param origDatasetLen: if samples is a combination of new and old datasets, set old dataset first with length 'origDatasetLen'
    :return: non-duplicate samples and/or indices of such samples
    """
    origDatasetLen = 0  # if there is no old dataset, take everything
    if oldDatasetPath is not None:
        dataset = np.load(oldDatasetPath, allow_pickle=True).item()["samples"]
        origDatasetLen = len(dataset)
        samples = np.concatenate((dataset, samples), axis=0)

    samplesTuple = [tuple(row) for row in samples]
    seen = set()
    seen_add = seen.add

    if returnInds:
        filtered = [
            [samplesTuple[i], i]
            for i in range(len(samplesTuple))
            if not (samplesTuple[i] in seen or seen_add(samplesTuple[i]))
        ]
        filteredSamples = [filtered[i][0] for i in range(len(filtered))]
        filteredInds = [filtered[i][1] for i in range(len(filtered))]

        return (
            np.asarray(filteredSamples[origDatasetLen:]),
            np.asarray(filteredInds[origDatasetLen:]) - origDatasetLen,
        )
    else:
        filteredSamples = [
            samplesTuple[i]
            for i in range(len(samplesTuple))
            if not (samplesTuple[i] in seen or seen_add(samplesTuple[i]))
        ]

        return np.asarray(filteredSamples[origDatasetLen:])

def calcAccuracy(preds,truths):
    acc=np.sum(np.equal(truths,preds))/len(truths)
    return acc 

def residCalc(preds,truths):
    residuals=abs((np.reshape(preds,len(preds))-np.reshape(truths,len(preds))))
    return residuals

def mseCalc(preds,truths):
    residuals=(np.reshape(preds,len(preds))-np.reshape(truths,len(preds)))
    resid_squared= [i**2 for i in residuals]
    new_sum=0 
    for i in range(0,len(resid_squared)):
        new_sum=new_sum+(resid_squared[i])
    return (new_sum/len(resid_squared))

def maeCalc(preds,truths):
    residuals=abs((np.reshape(preds,len(preds))-np.reshape(truths,len(preds))))
    new_sum=0 
    for i in range(0,len(residuals)):
        new_sum=new_sum+(residuals[i])
    return (new_sum/len(residuals))


def rmse(preds,truths):
    residuals=mseCalc(preds,truths)
    rmsee=np.sqrt(residuals)
    
    return rmsee