#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys
import numpy as np

 
CKPlusImagePath = '../data/Cohn-Kanade/CK+/extended-cohn-kanade-images/cohn-kanade-images/'
CKPlusNewImagePath = '../data/Cohn-Kanade/CK+/cropped_for_aufnet/'

INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1


def process():
    for SubIdx in range(1000):
        SubImagePath = CKPlusImagePath+'S'+str(SubIdx).zfill(3)+'/'
        # for each existed subject
        if os.path.isdir(SubImagePath):
            print(SubImagePath)
            # for each sequence
            for SeqIdx in range(20):
                left, top, right, bottom = 0.0, 0.0, 0.0, 0.0
                minx, miny, maxx, maxy = INT_MAX, INT_MAX, INT_MIN, INT_MIN
                SeqImagePath = SubImagePath+str(SeqIdx).zfill(3)+'/'
                print(SeqImagePath)
                if os.path.isdir(SeqImagePath):
                    for framename in os.listdir(SeqImagePath):
                    # detect face rect, return in [left, top, right, bottom]
                        if os.path.splitext(framename)[-1][1:] != 'png':
                            continue 
                        framepath = SeqImagePath + framename
                        print(framepath)
                        frame = cv2.imread(framepath)
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faceDetector = dlib.get_frontal_face_detector()
                        face = faceDetector(frame_gray, 1)
                        if len(face) == 0:
                            print("No face detected in frame{}".format(t))
                            continue

                        left, top, right, bottom = face[0].left(), face[0].top(), face[0].right(), face[0].bottom()
                        if left < minx:
                            minx = left
                        if top < miny:
                            miny = top
                        if right > maxx:
                            maxx = right
                        if bottom > maxy:
                            maxy = bottom

                    for framename in os.listdir(SeqImagePath):
                        if os.path.splitext(framename)[-1][1:] != 'png': 
                            continue
                        framepath = SeqImagePath + framename
                        saveImagefolder = CKPlusNewImagePath+'S'+str(SubIdx).zfill(3)+'/'+str(SeqIdx).zfill(3)
                        if not os.path.isdir(saveImagefolder):
                            os.makedirs(saveImagefolder)
                        saveImagePath = saveImagefolder + '/' + framename
                        print("saving ", saveImagePath)
                        frame = cv2.imread(framepath)
                        cv2.imwrite(saveImagePath,frame[miny: maxy, minx: maxx])




if __name__ == '__main__':
	process()


