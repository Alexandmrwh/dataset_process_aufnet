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

CKPlusNewImagePath = '../data/Cohn-Kanade/CK+/cropped_for_aufnet/'
CKPlusPickedLabels = '../data/Cohn-Kanade/CK+/CK+_Picked_aaai/'
CKPlus10folds = '../data/Cohn-Kanade/CK+/10folds/'

stride = 4
INT_MAX = sys.maxsize 
INT_MIN = -sys.maxsize-1
au_idx_lair = [1, 2, 4, 5, 6, 7, 9, 10, 12, 17, 23, 24, 25, 26, 43]
# au_idx_lair = [1, 2, 4, 5, 6, 7, 9, 12, 17, 23, 24, 25]
au_split_seq = [23, 9, 7, 5, 2, 6, 4, 1, 12, 17, 25, 10, 26, 43]

def split4flow():
    data = open(CKPlusNewImagePath+'finetuneflow'+'.txt','w')

    # for each subject
    for SubIdx in range(1000):
        SubImagePath = CKPlusNewImagePath+'S'+str(SubIdx).zfill(3)+'/'
        # for each existed subject
        if os.path.isdir(SubImagePath):
            # for each sequence
            for SeqIdx in range(20):
                framelist = []
                SeqImagePath = SubImagePath+str(SeqIdx).zfill(3)+'/'
                if os.path.isdir(SeqImagePath):

                    # for each image, save path to a list
                    for framename in os.listdir(SeqImagePath):
                        if os.path.splitext(framename)[-1][1:] != 'png':
                            continue
                        framelist.append(framename)
                
                framelist.sort()
                for i in range(0, len(framelist)-stride):
                    path = CKPlusNewImagePath+'S'+str(SubIdx).zfill(3)+'/'+str(SeqIdx).zfill(3)+'/'
                    frame1 = framelist[i]
                    frame2 = framelist[i+stride]
                    print(path + frame1, path + frame2, file = data)
    data.close()

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
                new_splitsres[au_idx_lair.index(au)][i] += audict[sub]

    return new_splits, new_splitsres

# generate 10 lists containing the subject id
def generate10splits(timestamp):
    metafile = np.loadtxt(CKPlusPickedLabels + 'CKPlus_meta.csv', dtype = np.int, delimiter = ',')
    print(metafile)

    aus = len(au_idx_lair)

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
        print("au {} appear in {} subs".format(au_idx_lair[i], len(aulist[i])))

    subdictlist = []
    for auidx in range(aus):
        sublist = aulist[auidx]
        appearslist = appeartimes[auidx]
        subdict = dict(zip(sublist, appearslist))
        subdictlist.append(subdict)

    splitres = np.zeros(shape=[aus+1,12], dtype=int)
    
    # split au24
    au24dict = subdictlist[au_idx_lair.index(24)]
    au24subs = list(au24dict.keys()) 
    au24splits = partition(au24subs, 10)
    for i in range(10):
        for sub in au24splits[i]:
            splitres[au_idx_lair.index(24)][i] += au24dict[sub]
    
    # split the rest
    for idx in au_split_seq:
        if idx == 23:
            ausplits_new, splitres_new = splitnew(au24splits, 23, subdictlist[au_idx_lair.index(23)], splitres)
        else: 
            ausplits_new, splitres_new = splitnew(ausplits_new, idx, subdictlist[au_idx_lair.index(idx)], splitres_new)
    
    splitres_new[aus] = splitres_new.sum(axis=0)
    for i in range(aus):
        splitres_new[i][11] = sum(splitres_new[i][:])
        splitres_new[i][10] = sum(appeartimes[i])            

    logfile = open(CKPlus10folds+ timestamp + '10fold-split.txt', 'a+')
    print(splitres_new, ausplits_new, file=logfile)
    logfile.close()

    print(ausplits_new)
    print(splitres_new)

    return ausplits_new

# generate one txt for each fold
def generate10folds(timestamp, ausplits, stride):

    # for each fold, open log txt
    numfold = len(ausplits)
    for foldidx in range(numfold):
        foldtxt = open(CKPlus10folds + timestamp +str(foldidx)+'.txt', 'w')
        subs = ausplits[foldidx]
        # for each subject
        for sub in subs:
            SubImagePath = CKPlusNewImagePath+'S'+str(sub).zfill(3)+'/'
            if os.path.isdir(SubImagePath):
                # prepare labels of all sequence of this sub
                squencelabel = {}
                sublabelpath = CKPlusPickedLabels + 'S' + str(sub).zfill(3) + '.txt'
                sublabel = open(sublabelpath, 'r')
                for _, lines in enumerate(sublabel.readlines()):
                    au_pos_tmp = np.zeros(len(au_idx_lair))
                    squencepath, au_pos_tmp[0], au_pos_tmp[1], \
                        au_pos_tmp[2], au_pos_tmp[3], \
                        au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
                        au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9], \
                        au_pos_tmp[10], au_pos_tmp[11], au_pos_tmp[12],\
                        au_pos_tmp[13], au_pos_tmp[14] = lines.split()
                    squenceau = ndarray2string(au_pos_tmp)
                    squenceid = squencepath[-4: -1]
                    squencelabel[squenceid] = squenceau
                
                # for each sequence
                for SeqIdx in range(20):
                    framelist = []
                    SeqImagePath = SubImagePath+str(SeqIdx).zfill(3)+'/'
                    if os.path.isdir(SeqImagePath):
                        # for each image, save path to a list
                        for framename in os.listdir(SeqImagePath):
                            if os.path.splitext(framename)[-1][1:] != 'png':
                                continue
                            framelist.append(framename)
                    framelist.sort()
                    for i in range(0, len(framelist)-stride+1, stride):
                        path = CKPlusNewImagePath+'S'+str(sub).zfill(3)+'/'+str(SeqIdx).zfill(3)+'/'
                        frame1 = framelist[i]
                        for j in range(1, stride):
                            frame2 = framelist[i+j]
                            label = squencelabel[str(SeqIdx).zfill(3)]
                            print(path + frame1, path + frame2, label, file=foldtxt)

        foldtxt.close()

def generatetrainvaltxt(timestamp, valfold):
    traintxt = open(CKPlus10folds + timestamp +'train_e'+ str(valfold) + '.txt', 'w')
    valtxt = open(CKPlus10folds + timestamp +'val_'+ str(valfold) + '.txt', 'w')

    valfoldtxt = open(CKPlus10folds + timestamp + str(valfold)+'.txt')
    for t, lines in enumerate(valfoldtxt.readlines()):
        print(lines, file=valtxt)
    
    all = [x for x in range(10)]
    trainset = list((set(all)) - set([valfold]))
    for foldidx in trainset:
        foldtxt = open(CKPlus10folds + timestamp + str(foldidx) + '.txt')
        for t, lines in enumerate(foldtxt.readlines()):
            print(lines, file=traintxt)

if __name__ == "__main__":
    # split4flow()
    timestamp = str(int((time.time()))) + '/'
    os.makedirs(CKPlus10folds + timestamp)

    ausplits = generate10splits(timestamp)
    generate10folds(timestamp, ausplits, stride)
    generatetrainvaltxt(timestamp, valfold=0)
    