#!/usr/bin/env python
#coding=utf-8
import os
import cv2
import csv
import dlib
import random
import sys
import time
import numpy as np
from ckplus_label_process import ndarray2string

au_idx = [1, 2, 4, 5, 6, 9 ,12, 17, 25, 26]
au_split_seq = [9, 2, 1, 17, 6, 4, 26, 12, 25]
DisfaAllLabelPath = "../data/DISFA/DISFA_all_label/"
Disfa10FoldsPath = "../data/DISFA/"

stride = 2 # missing rate = 0.5
# stride = 3 # missing rate = 0.33
# stride = 4 # missing rate = 0.25
# stride = 5 # missing rate = 0.2

def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def splitnew(splits_in, au, audict, splitres):
    new_splits = splits_in
    new_splitsres = splitres
    ausubs = list(audict.keys())
    splits_set = set.union(*map(set,splits_in))
    ausubs_res = list(set(ausubs) - set(splits_set))
    aures_splits = partition(ausubs_res, 10)
    for i in range(10):
        for sub in aures_splits[i]:
            new_splits[i].append(sub)
    
    for i in range(10):
        for sub in new_splits[i]:
            if sub in ausubs:
                new_splitsres[au_idx.index(au)][i] += audict[sub]

    return new_splits, new_splitsres

def generate10splits(expname):
    metafile = np.loadtxt(DisfaAllLabelPath + 'DISFA_meta.csv', dtype = np.int, delimiter = ',')
    print(metafile)

    aus = len(au_idx)

    aulist = [] # for each au, which subject contains this au
    appeartimes = []
    for auidx in range(1, aus+1):
        aulisttmp, appeartimestmp = [], []
        for i in range(metafile.shape[0]):
            if metafile[i][auidx] != 0:
                aulisttmp.append(metafile[i][0])
                appeartimestmp.append(metafile[i][auidx])
        aulist.append(aulisttmp)
        appeartimes.append(appeartimestmp)
    
    for i in range(len(aulist)):
        print("au {} appear in {} subs".format(au_idx[i], len(aulist[i])))

    subdictlist = []
    for auidx in range(aus):
        sublist = aulist[auidx]
        appearslist = appeartimes[auidx]
        subdict = dict(zip(sublist, appearslist))
        subdictlist.append(subdict)

    splitres = np.zeros(shape=[aus+1, 12], dtype=int)
    
    # split au5
    au5dict = subdictlist[au_idx.index(5)]
    au5subs = list(au5dict.keys()) 
    au5splits = partition(au5subs, 10)
    for i in range(10):
        for sub in au5splits[i]:
            splitres[au_idx.index(5)][i] += au5dict[sub]
    
    # split the rest
    for idx in au_split_seq:
        if idx == 9:
            ausplits_new, splitres_new = splitnew(au5splits, 9, subdictlist[au_idx.index(9)], splitres)
        else: 
            ausplits_new, splitres_new = splitnew(ausplits_new, idx, subdictlist[au_idx.index(idx)], splitres_new)
    
    splitres_new[aus] = splitres_new.sum(axis=0)
    for i in range(aus):
        splitres_new[i][11] = sum(splitres_new[i][:])
        splitres_new[i][10] = sum(appeartimes[i])  
    
    print(ausplits_new)
    print(splitres_new)
    logfile = open(Disfa10FoldsPath + expname + '10fold-split.txt', 'a+')
    print(splitres_new, ausplits_new, file=logfile)
    logfile.close()

    return ausplits_new

def generate10folds(expname, ausplits, stride):
    # for each fold, open log txt
    numfold = len(ausplits)
    for foldidx in range(numfold):
        foldtxt = open(Disfa10FoldsPath + expname +str(foldidx)+'.txt', 'w')
        sessions = ausplits[foldidx]
        print("=====> fold", foldidx, sessions)
        # for each session
        for session_idx in sessions:
            txt_name = str(session_idx).zfill(2)
            session_label_path = DisfaAllLabelPath + "disfa_10aus_session_{}.txt".format(txt_name)
            print(session_label_path)
            if os.path.isfile(session_label_path):
                session_label = open(session_label_path,'r')
                all_samples = session_label.readlines()
                sample_num = len(all_samples)
                aus = np.zeros(10)
                for i in range(0, sample_num-stride+1, stride):
                    line1 = all_samples[i]
                    frameIdx1, aus[0], aus[1], aus[2], aus[3], \
                            aus[4], aus[5], aus[6], aus[7], aus[8], aus[9]= line1.split()
                    for j in range(1, stride):
                        aus = aus.astype(np.int)
                        line2 = all_samples[i+j]
                        frameIdx2, _, _, _, _, \
                                _, _, _, _, _, _, = line2.split()
                        print(frameIdx1, frameIdx2, aus[0], aus[1], aus[2], aus[3], \
                                aus[4], aus[5], aus[6], aus[7], aus[8], aus[9], file=foldtxt)
        foldtxt.close()

def generatetrainvaltxt(expname, valfold):
    traintxt = open(Disfa10FoldsPath + expname + "fold" + str(i) +'/' + 'train.txt', 'w')
    valtxt = open(Disfa10FoldsPath + expname + "fold" + str(i) +'/' + 'val.txt', 'w')

    valset = open(Disfa10FoldsPath + expname + str(valfold)+'.txt')
    for _, lines in enumerate(valset.readlines()):
        print(lines.strip('\n'), file=valtxt)

    all = [x for x in range(10)]
    trainset = list((set(all)) - set([valfold]))
    for foldidx in trainset:
        foldtxt = open(Disfa10FoldsPath + expname + str(foldidx) + '.txt')
        for _, lines in enumerate(foldtxt.readlines()):
            print(lines.strip('\n'), file=traintxt)


if __name__ == '__main__':
    expnum = "EXP" + str(1) + '/'
    expname = str(expnum)
    os.makedirs(Disfa10FoldsPath + expname)

    ausplits = generate10splits(expname)
    generate10folds(expname, ausplits, stride)
    for i in range(10):
        os.makedirs(Disfa10FoldsPath + expname + "fold" + str(i) +'/')
        generatetrainvaltxt(expname, valfold=i)
    




    
