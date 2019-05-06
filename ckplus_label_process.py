#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys

CKPlusAllLabels = '../data/Cohn-Kanade/CK+/CK+_all_label/'
CKPlusImagePath = '../data/Cohn-Kanade/CK+/extended-cohn-kanade-images/cohn-kanade-images/'
CKPlusLabelPath = '../data/Cohn-Kanade/CK+/FACS_labels/FACS/' 
INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1
au_idx = [1, 2, 4, 5, 6, 7, 9 ,10 ,11, 12, 
            13, 14, 15, 16, 17, 18, 20, 21, 23 , 24, 
            25 ,26 , 27, 28, 29, 31, 34, 38, 39, 43]

def label_process():
    # for each subject, generate a label file recording each sequence and its label
    for SubIdx in range(1000):
        newLabel = open(CKPlusAllLabels+'S'+str(SubIdx).zfill(3)+'.txt', 'w')
        SubLabelPath = CKPlusLabelPath+'S'+str(SubIdx).zfill(3)+'/'
        if os.path.isdir(SubLabelPath):
            print('=====> processing subject: {}'.format(SubIdx))
            # for each sequence
            for SeqIdx in range(20):
                _SeqLabelPath = SubLabelPath+str(SeqIdx).zfill(3)+'/'
                if os.path.isdir(_SeqLabelPath):
                    for txtName in os.listdir(_SeqLabelPath):
                        SeqLabelPath = _SeqLabelPath+txtName
                        SeqLabel = open(SeqLabelPath, 'r')
                        for _, lines in enumerate(SeqLabel.readlines()):
                            au, intensity = lines.split()
                            au = int(au)
                            intensity = int(intensity)
                            
                            

                    
                    










if __name__ == '__main__':
    label_process()


