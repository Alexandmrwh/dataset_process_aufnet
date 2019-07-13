#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys
import numpy as np

stride = 3
 
CKPlusNewImagePath = '../data/Cohn-Kanade/CK+/cropped_for_aufnet/'

INT_MAX = sys.maxsize 
INT_MIN = -sys.maxsize-1

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
                    
                    for i in range(0, len(framelist)-stride, stride):
                        frame1 = framelist[i]
                        frame2 = framelist[i+stride]
                        print(frame1, frame2, file = data)
    data.close()

if __name__ == "__main__":
    split()
    