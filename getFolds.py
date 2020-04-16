import numpy as np
import random
import csv
from utils import readCsv, writeCsv
import click


def getFold(fold = 0, fname_in = 'trainFolds.csv',
            fnames = ['CTs.csv','Fleischner.csv','Nodules.csv'],
            prefix_in = 'train', prefix_out = '',
            excludeFold = True):
    
    if not prefix_out:
        prefix_out = 'fold{}'.format(fold) # eg. fold0
    
    #Get fold lnds
    nodules = readCsv(fname_in)
    header = nodules[0]
    lines = nodules[1:]
    
    foldind = header.index('Fold{}'.format(fold)) # get fold idx from file
    foldlnd = [l[foldind] for l in lines if len(l)>foldind] # select correct lnd number except with missing data

    for fname in fnames: # loop thru filetypes 
        lines = readCsv(prefix_in+fname) 
        header = lines[0]
        lines = lines[1:]
        
        lndind = header.index('LNDbID')
        if not excludeFold:
            lines = [l for l in lines if l[lndind] in foldlnd]
        else:
            lines = [l for l in lines if not l[lndind] in foldlnd]
        
        #Save to csv
        writeCsv(prefix_out+fname,[header]+lines)
            
if __name__ == "__main__":
    # Get all folds except 0 from trainset
    getFold(fold=0, fnames=['Nodules_gt.csv']) # want consolidated 
