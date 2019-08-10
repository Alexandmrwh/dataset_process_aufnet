#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import random
import sys
import numpy as np

CKPlusNewImagePath = '../data/Cohn-Kanade/CK+/cropped_for_aufnet/'
CKPlusPickedLabels = '../data/Cohn-Kanade/CK+/CK+_Picked_aaai/'

stride = 4
INT_MAX = sys.maxsize 
INT_MIN = -sys.maxsize-1
au_idx_lair = [1, 2, 4, 5, 6, 7, 9, 12, 17, 23, 24, 25]
au_split_seq = [23, 9, 7, 5, 2, 6, 4, 1, 12, 17, 25]

def split():
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

def split10():
    metafile = np.loadtxt(CKPlusPickedLabels + 'CKPlus_meta.csv', dtype = np.int, delimiter = ',')
    print(metafile)

    aulist = [] # for each au, which subject contains this au
    appeartimes = []
    for auidx in range(1, 13):
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
    for auidx in range(12):
        sublist = aulist[auidx]
        appearslist = appeartimes[auidx]
        subdict = dict(zip(sublist, appearslist))
        subdictlist.append(subdict)

    splitres = np.zeros(shape=[13,12], dtype=int)
    
    # split 24
    au24dict = subdictlist[au_idx_lair.index(24)]
    au24subs = list(au24dict.keys()) 
    au24splits = partition(au24subs, 10)
    for i in range(10):
        for sub in au24splits[i]:
            splitres[au_idx_lair.index(24)][i] += au24dict[sub]
    
    for idx in au_split_seq:
        if idx == 23:
            ausplits_new, splitres_new = splitnew(au24splits, 23, subdictlist[au_idx_lair.index(23)], splitres)
        else: 
            ausplits_new, splitres_new = splitnew(ausplits_new, idx, subdictlist[au_idx_lair.index(idx)], splitres_new)
    
    splitres_new[12] = splitres_new.sum(axis=0)
    for i in range(12):
        splitres_new[i][11] = sum(splitres_new[i][:])
        splitres_new[i][10] = sum(appeartimes[i])            

    print(splitres_new)
    print(ausplits_new)
    
if __name__ == "__main__":
    # split()
    split10()
    