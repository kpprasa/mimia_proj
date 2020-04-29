''' getFolds.py

    Creates k-folds csv from training data csv. 
    If excludeFold=True, then this is for training (ie others are consolidated)
    else for validation. 
    From LNDb challenge. Modified to fit needs of generation. 

    ______________________________________________________
    by Kiran Prasad <kiranpra@cs.cmu.edu>
    16-725 Methods in Medical Image Analysis Final Project
    ======================================================
'''
import numpy as np
import random
import csv
from utils import readCsv, writeCsv
import argparse


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


parser = argparse.ArgumentParser(
    description="Adversarial Training for LNDb Dataset; MIMIA 2020 Final project by Kiran Prasad"
)
parser.add_argument(
    "--val",
    type=bool,
    default=True,
    help="Generate Validation sets or training sets",
)
args = parser.parse_args()
if __name__ == "__main__":
    # Get all folds except fold from trainset
    fold = input('fold number?')
    val = args.val
    getFold(fold=fold, fnames=['Nodules_gt.csv'], prefix_out='val_fold_{}'.format(fold) if val else 'folds_exclude_{}'.format(fold), excludeFold=False) # want consolidated 
