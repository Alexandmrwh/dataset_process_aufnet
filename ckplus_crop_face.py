#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import csv
import dlib
import sys
import math
import numpy as np
from imutils import face_utils

from disfa_crop_face import get_facelandmark, alignment

CKPlusImagePath = '../data/Cohn-Kanade/CK+/extended-cohn-kanade-images/cohn-kanade-images/'
CKPlusNewImagePath = '../data/Cohn-Kanade/CK+/cropped_for_aufnet/'
CKPlusAlignedImagePath = '../data/Cohn-Kanade/CK+/align_CK+/'
CKPlusAlignedCropImagePath = '../data/Cohn-Kanade/CK+/align_crop_CK+/'

shapePredictorPath = '../../shape_predictor_68_face_landmarks.dat'

faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)

INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1

def align_crop_video(print_every=200):
    for SubIdx in range(1000):
        SubImagePath = CKPlusImagePath+'S'+str(SubIdx).zfill(3)+'/'
        AlignedImagePath = CKPlusAlignedImagePath+'S'+str(SubIdx).zfill(3)+'/'
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
                        # for each frame
                        if os.path.splitext(framename)[-1][1:] != 'png':
                            continue 
                        framepath = SeqImagePath + framename
                        frame = cv2.imread(framepath)
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # align and save
                        featureList, _, _ = get_facelandmark(frame_gray)
                        if featureList is not None and len(featureList):
                            alignimg, _ = alignment(frame, featureList)

                            saveAlignedImageFolder = CKPlusAlignedImagePath+'S'+str(SubIdx).zfill(3)+'/'+str(SeqIdx).zfill(3)
                            if not os.path.isdir(saveAlignedImageFolder):
                                os.makedirs(saveAlignedImageFolder)
                            saveImagePath = saveImagefolder + '/' + framename
                            print("saving ", saveImagePath)
                            cv2.imwrite(saveImagePath, alignimg)

                        # detect face in aligned image
                        faceDetector = dlib.get_frontal_face_detector()
                        face = faceDetector(alignimg, 1)
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

                    # crop
                    SeqImagePath = CKPlusAlignedImagePath+str(SeqIdx).zfill(3)+'/'
                    for framename in os.listdir(SeqImagePath):
                        if os.path.splitext(framename)[-1][1:] != 'png': 
                            continue
                        framepath = SeqImagePath + framename
                        saveImagefolder = CKPlusAlignedCropImagePath+'S'+str(SubIdx).zfill(3)+'/'+str(SeqIdx).zfill(3)
                        if not os.path.isdir(saveImagefolder):
                            os.makedirs(saveImagefolder)
                        saveImagePath = saveImagefolder + '/' + framename
                        print("saving ", saveImagePath)
                        
                        width = maxx - minx
                        heitht = maxy - miny
                        frame = cv2.imread(framepath)
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        featureList, x, y = get_facelandmark(frame_gray)
                        
                        if featureList is not None and len(featureList) > 45:
                            Xs = featureList[::2]
                            Ys = featureList[1::2]
                            eye_center =((Xs[36] + Xs[45]) * 1./2, (Ys[36] + Ys[45]) * 1./2)

                            left = int(eye_center[0] - 0.5 * width)
                            right = int(eye_center[0] + 0.5 * width)
                            top = int(eye_center[1] - 0.4 * height)
                            bottom = int(eye_center[1] + 0.6 * height)
                            cv2.imwrite(savepath,frame[top: bottom, left: right])
                            print(savepath)


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
	# process()
    align_crop_video()


