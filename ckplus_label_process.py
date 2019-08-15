#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys
import numpy as np

CKPlusAllLabels = '../data/Cohn-Kanade/CK+/CK+_all_label/'
CKPlusPickedLabelsSave = '../data/Cohn-Kanade/CK+/CK+_Picked_aaai/'
CKPlusPickedLabels = '../data/Cohn-Kanade/CK+/CK+_Picked_label/'
# CKPlusImagePath = '../data/Cohn-Kanade/CK+/extended-cohn-kanade-images/cohn-kanade-images/'
CKPlusImagePath = '../data/Cohn-Kanade/CK+/cropped_for_aufnet/'
CKPlusLabelPath = '../data/Cohn-Kanade/CK+/FACS_labels/FACS/'
CKPlusPickedImagesTxt = '../data/Cohn-Kanade/CK+/PickedImagesTxt/'

INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1
au_idx = [1, 2, 4, 5, 6, 7, 9 , 10, 11, 12, 
            13, 14, 15, 16, 17, 18, 20, 21, 22, 23 , 24, 
            25 ,26 , 27, 28, 29, 30, 31, 34, 38, 39, 43, 44, 45, 54, 61, 62, 63, 64]

au_idx_lair = [1, 2, 4, 5, 6, 7, 9, 10, 12,
               17, 23, 24, 25, 26, 43]

# au_idx_lair = [1, 2, 4, 5, 6, 7, 9, 12, 17, 23, 24, 25]

def ndarray2string(label):
    label = label.astype(int)
    label_str = ' '
    for num in label:
        if num == 1:
            label_str += ' 1'
        elif num == 0:
            label_str += ' 0'
    return label_str

def all_label_process():
    for SubIdx in range(1000):
        SubLabelPath = CKPlusLabelPath+'S'+str(SubIdx).zfill(3)+'/'
        SubImagePath = CKPlusImagePath+'S'+str(SubIdx).zfill(3)+'/'

        # for each existed subject, generate a label file recording each sequence and its label
        if os.path.isdir(SubLabelPath):
            SubNewLabel = open(CKPlusAllLabels+'S'+str(SubIdx).zfill(3)+'.txt', 'w')
            print('=====> processing subject: {}, results writing to: {}'.format(SubIdx, SubNewLabel))
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
                    print(SeqImagePath, onehotLabelStr, file=SubNewLabel)
            SubNewLabel.close()
                            
def picked_label_process():
    for SubIdx in range(1000):
        SubLabelPath = CKPlusLabelPath+'S'+str(SubIdx).zfill(3)+'/'
        SubImagePath = CKPlusImagePath+'S'+str(SubIdx).zfill(3)+'/'
        # for each existed subject, generate a label file recording each sequence and its label
        if os.path.isdir(SubLabelPath):
            SubNewLabel = open(CKPlusPickedLabelsSave+'S'+str(SubIdx).zfill(3)+'.txt', 'w')
            print('=====> processing subject: {}, results writing to: {}'.format(SubIdx, SubNewLabel))
            # for each sequence, store the path to the sequence and its label to the new label file of the subject
            for SeqIdx in range(20):
                _SeqLabelPath = SubLabelPath+str(SeqIdx).zfill(3)+'/'
                SeqImagePath = SubImagePath+str(SeqIdx).zfill(3)+'/'
                if os.path.isdir(_SeqLabelPath):
                    onehotLabel = np.zeros(len(au_idx_lair))
                    for txtName in os.listdir(_SeqLabelPath):
                        SeqLabelPath = _SeqLabelPath+txtName
                        SeqLabel = open(SeqLabelPath, 'r')
                        for _, lines in enumerate(SeqLabel.readlines()):
                            au, intensity = lines.split()
                            au = int(float(au))
                            if au in au_idx_lair:
                                # intensity = int(float(intensity))
                                onehotLabel[au_idx_lair.index(au)] = 1
                        onehotLabelStr = ndarray2string(onehotLabel)
                        print(SeqImagePath, onehotLabelStr, file=SubNewLabel)
            SubNewLabel.close()

def pick_images_with_au():
    for SubIdx in range(1000):
        SubLabelPath = CKPlusLabelPath+'S'+str(SubIdx).zfill(3)+'/'
        SubImagePath = CKPlusImagePath+'S'+str(SubIdx).zfill(3)+'/'
        # for each existed subject, generate a label file recording each sequence and its label
        if os.path.isdir(SubLabelPath):
            # for each sequence, store the path to the sequence and its label to the new label file of the subject
            for SeqIdx in range(20):
                _SeqLabelPath = SubLabelPath+str(SeqIdx).zfill(3)+'/'
                SeqImagePath = SubImagePath+str(SeqIdx).zfill(3)+'/'
                if os.path.isdir(_SeqLabelPath):
                    if not os.path.isdir(CKPlusPickedImagesTxt+'S'+str(SubIdx).zfill(3)+'/'):
                        os.makedirs(CKPlusPickedImagesTxt+'S'+str(SubIdx).zfill(3)+'/')
                    SubPickedImage = open(CKPlusPickedImagesTxt+'S'+str(SubIdx).zfill(3)+'/'+str(SeqIdx).zfill(3)+'.txt', 'w')
                    imagename = os.listdir(SeqImagePath)
                    imagename.sort()
                    imagelast = imagename[-1]
                    imagenum = int(imagelast[15]) * 10 + int(imagelast[16])
                    if imagenum > 18:
                        nums = 10
                    else:
                        nums = 5
                    imageprefix = "S"+str(SubIdx).zfill(3)+"_"+str(SeqIdx).zfill(3)+"_"
                    for imageindex in range(imagenum-nums+1, imagenum+1):
                        picked_image = SeqImagePath+imageprefix+str(imageindex).zfill(8)+'.png'
                        print(picked_image, file=SubPickedImage)

            SubPickedImage.close()


if __name__ == '__main__':
    # all_label_process()
    # picked_label_process()
    pick_images_with_au()


