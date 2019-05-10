#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys
import numpy as np

CKPlusAllLabels = '../data/Cohn-Kanade/CK+/CK+_all_label/'
CKPlusPickedLabels = '../data/Cohn-Kanade/CK+/CK+_Picked_label/'
CKPlusImagePath = '../data/Cohn-Kanade/CK+/extended-cohn-kanade-images/cohn-kanade-images/'
CKPlusLabelPath = '../data/Cohn-Kanade/CK+/FACS_labels/FACS/' 
INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1
au_idx = [1, 2, 4, 5, 6, 7, 9 , 10, 11, 12, 
            13, 14, 15, 16, 17, 18, 20, 21, 22, 23 , 24, 
            25 ,26 , 27, 28, 29, 30, 31, 34, 38, 39, 43, 44, 45, 54, 61, 62, 63, 64]

au_idx_lair = [1, 2, 4, 5, 6, 7, 9, 10, 12,
				17, 23, 24, 25, 26, 43]
