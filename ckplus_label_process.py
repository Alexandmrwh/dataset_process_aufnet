#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys
import numpy as np

CKPlusAllLabels = '../data/Cohn-Kanade/CK+/CK+_all_label/'
CKPlusImagePath = '../data/Cohn-Kanade/CK+/extended-cohn-kanade-images/cohn-kanade-images/'
CKPlusLabelPath = '../data/Cohn-Kanade/CK+/FACS_labels/FACS/' 
INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1
au_idx = [1, 2, 4, 5, 6, 7, 9 ,10 ,11, 12, 
            13, 14, 15, 16, 17, 18, 20, 21, 22, 23 , 24, 
            25 ,26 , 27, 28, 29, 31, 34, 38, 39, 43, 45, 62]

def ndarray2string(label):
    label = label.astype(int)
    label_str = ' '
    for num in label:
        if num == 1:
            label_str += ' 1'
        elif num == 0:
            label_str += ' 0'
    return label_str

def label_process():
    # for each subject, generate a label file recording each sequence and its label
    for SubIdx in range(1000):
        newLabel = open(CKPlusAllLabels+'S'+str(SubIdx).zfill(3)+'.txt', 'w')
        SubLabelPath = CKPlusLabelPath+'S'+str(SubIdx).zfill(3)+'/'
        SubImagePath = CKPlusImagePath+'S'+str(SubIdx).zfill(3)+'/'
        if os.path.isdir(SubLabelPath):
            print('=====> processing subject: {}'.format(SubIdx))
            # for each sequence, store the path to the sequence and its label to the new label file of the subject
            for SeqIdx in range(20):
                _SeqLabelPath = SubLabelPath+str(SeqIdx).zfill(3)+'/'
                SeqImagePath = SubImagePath+str(SeqIdx).zfill(3)+'/'
                if os.path.isdir(_SeqLabelPath):
                    onehotLabel = np.zeros(len(au_idx))
                    for txtName in os.listdir(_SeqLabelPath):
                        SeqLabelPath = _SeqLabelPath+txtName
                        SeqLabel = open(SeqLabelPath, 'r')
                        for _, lines in enumerate(SeqLabel.readlines()):
                            au, intensity = lines.split()
                            au = int(float(au))
                            # intensity = int(float(intensity))
                            onehotLabel[au_idx.index(au)] = 1
                            onehotLabelStr = ndarray2string(onehotLabel)
                    print(SeqImagePath, onehotLabelStr)
                            
                            
                            
                            






if __name__ == '__main__':
    label_process()


