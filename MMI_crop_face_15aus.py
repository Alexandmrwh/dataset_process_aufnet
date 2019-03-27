from __future__ import print_function
import torch 
import numpy
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.models as models
import torch.utils.data 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import hamming_loss
from PIL import Image
import copy
import numpy as np
import sys, getopt
import time
import os
import os.path

import time

print(torch.__version__)


dataset_dir = ''

input_size = 64

num_classes = 6

batch_size = 64

learning_rate = 0.003

num_epochs = 1000

print_every = 10

feature_extract = True

from cv2 import WINDOW_NORMAL
import sys
import cv2
import glob 
import time
import os
import math
from sklearn.externals import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import svm
import numpy as np
from PIL import Image
import threading
from sklearn.decomposition import FastICA
import xml.dom.minidom
subInfo = [[1,120],[119,245],[245,351],[351,471],[471,580],[580,690],[689,795],[793,860],[859,965],[964,1067],
[1067,1142],[1142,1215],[1215,1287],[1287,1416],[1416,1500],[1500,1569]]

left_right = [2,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0] # 1 represent the face is in the right half picture 0 left

subject_train = [1,2,3,4,5,6,9,14] 
subject_val = [7,8,15,16]
subject_test = [10,11,12,13]

AU = [1,2,4,5,6,7,9,10,12,17,23,24,25,26,43]
Pos = np.zeros([17,len(AU)])
Neg = np.zeros([17,len(AU)])
path_MMI = "/home/ubuntu/AU-WorkSite/FAC_DataBase/MMI_cropped/"

faceDet = cv2.CascadeClassifier("/home/ubuntu/AU-WorkSite/Expri11.30/haarcascades/haarcascade_frontalface_default.xml")

def _locate_faces(image):
    faces = faceDet.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )
    return faces  # list of (x, y, w, h)

def produce_data(SavePath, SubNum, subDown, subUp, status):
    print ("Subject: ", SubNum)
    ReadPath = "/home/ubuntu/AU-WorkSite/FAC_DataBase/MMI database/subject"+str(SubNum)+"/Sessions/"
    SavePath = SavePath + status + "/"
    if not os.path.isdir(SavePath):
        os.makedirs(SavePath)
    
    for SessionNum in range(subDown,subUp):
        print("    ", SessionNum)
        if(SessionNum == 244): continue
        SessionPath = ReadPath + str(SessionNum)+"/"
        if not os.path.isdir(SessionPath):
            continue
        num1 = str(SubNum).zfill(3)
        num2 = str(SessionNum-subDown+1).zfill(3)
        VideoPath = SessionPath + "S"+num1+"-"+num2+".avi"
        LabelPath = SessionPath + "S"+num1+"-"+num2+".xml"

        #deal with label
        if not os.path.isfile(LabelPath):
            print ("Session ", SessionNum, "subject", SubNum, "xml missing!")
            continue;

        dom  = xml.dom.minidom.parse(LabelPath)
        root = dom.documentElement
        #FrameNum = root.getAttribute('NumFrames')
        ty = []
        ActionUnit = root.getElementsByTagName('ActionUnit')
        for au in ActionUnit:
            ty.append(au.getAttribute('Number')) # ty contains the au numbers that occured in the avi
        #print(ty)
        for index in range(len(AU)):
            if str(AU[index]) in ty: # 4 here means au4
                print ("Session", SessionNum, "labeled with au",AU[index])
            else:
                #print "Session", SessionNum, "passed with no au4"
                continue
            #count
            mXs = []
            mYs = []
            vc = cv2.VideoCapture(VideoPath)
            read_value, webcam_image = vc.read()
            FrameNum = 0
            while read_value:
                FrameNum = FrameNum+1
                faces = _locate_faces(webcam_image)
                if len(faces) is not 0:
                    x, y, w, h = faces[0]
                    mXs.append(x+w/2)
                    mYs.append(y+h/2)
                read_value, webcam_image = vc.read()
            if len(mXs) is not 0 and len(mYs) is not 0:
                Mx = sum(mXs)/len(mXs)
                My = sum(mYs)/len(mYs)
            else:
                continue

            #crop
            vc = cv2.VideoCapture(VideoPath)
            read_value, webcam_image = vc.read()
            cur_frame = 0
            i = 1

            image = Image.fromarray(webcam_image)
            width, height = image.size[:2]
            #print(width,height)
            box_1 = (0,0,width/2,height)
            box_2 = (width/2,0,width,height)
            #print(box_1,box_2)
            SavePath4 = ''

            while read_value:
                X = []
                cur_frame += 1  

                if(cur_frame == FrameNum/3):
                    '''
                    image = Image.fromarray(webcam_image)
                    if(left_right[SubNum] == 0):
                        image = image.crop(box_1)
                    else: image = image.crop(box_2)

                    wtmp, htmp = image.size[:2]
                    box = (0, My-wtmp/2, wtmp, My+wtmp/2)
                    image = image.crop(box)

                    image1 = np.array(image)
                    '''
                    SavePath4 = SavePath+"sub"+str(SubNum)+"Session"+str(SessionNum)+'_'+str(i)+".png"
                    #cv2.imwrite(SavePath4, image1)

                if(cur_frame > FrameNum/3 and cur_frame < FrameNum *2 /3):
                    '''
                    image2 = Image.fromarray(webcam_image)
                    if(left_right[SubNum] == 0):
                        image2 = image2.crop(box_1)
                    else: image2 = image2.crop(box_2)
                    
                    wtmp, htmp = image2.size[:2]
                    box = (0, My-wtmp/2, wtmp, My+wtmp/2)
                    image2 = image2.crop(box)

                    image2 = np.array(image2)
                    '''
                    SavePath5 = SavePath+"sub"+str(SubNum)+"Session"+str(SessionNum)+'_'+str(i)+".png"
                    #cv2.imwrite(SavePath5,image2)

                    res = []
                    for j in range(len(AU)):
                        if str(AU[j]) in ty:
                            res.append(1)
                            Pos[SubNum][j] += 1
                        else: 
                            res.append(0)
                            Neg[SubNum][j] += 1
                    
                    if(SavePath4!=''):
                        if(SubNum in subject_val):
                            print(str(SavePath4),str(SavePath5),end = ' ',file = fv) 
                            for i in range(len(AU)):
                                print(res[i], end = ' ',file = fv)
                            print("",file = fv)
                        elif(SubNum in subject_test):
                            print(str(SavePath4),str(SavePath5),end = ' ',file = ftest) 
                            for i in range(len(AU)):
                                print(res[i], end = ' ',file = ftest)
                            print("",file = ftest)
                        else:
                            print(str(SavePath4),str(SavePath5),end = ' ',file = ft)
                            for i in range(len(AU)):
                                print(res[i], end = ' ',file = ft)
                            print("",file = ft)
                    i += 1
                    SavePath4 = SavePath5
                read_value, webcam_image = vc.read()

def count_session(SubNum, subDown, subUp):
    ReadPath1 = "/home/ubuntu/AU-WorkSite/FAC_DataBase/MMI database/subject"+str(SubNum)+"/Sessions/"
    SavePath2 = "/home/ubuntu/AU-WorkSite/FAC_DataBase/MMI database/subject"+str(SubNum)+"/Sessions/" 
    AU4Session = []
    DetectedAU4Session = []
    count = 0

    for SessionNum in range(subDown,subUp):
        if(SessionNum == 244): continue
        SessionPath = ReadPath1 + str(SessionNum)+"/"
        if not os.path.isdir(SessionPath):
            continue
        else:
            count += 1
    return count

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root,datatxt, replicates = 1,transform = transforms.ToTensor(),target_transform = None):
        fh = open(root + datatxt,'r')
        imgs = []
        i = 0
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            img_path = [words[0],words[1]]
            res = []
            for index in range(2, num_classes + 2):
                res.append(int(words[index]))
                    #print(i)
                imgs.append((img_path, res))
                i += 1
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #print(label)
        img1 = Image.open(fn[0]).convert('RGB')
        img2 = Image.open(fn[1]).convert('RGB')
        label = np.array(label)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def __len__(self):
        return len(self.imgs)

if (__name__ == '__main__'):
    SavePath = path_MMI
    ff = open(SavePath + 'pos_neg.txt','w')
    f = open(SavePath+'out.txt','w')
    ft = open(SavePath + 'train.txt','w')
    fv = open(SavePath + 'val.txt','w')
    ftest = open(SavePath+'test.txt','w')

    # for i in range(1, 17):
    #     print(i, count_session(i, subInfo[i-1][0], subInfo[i-1][1]))
    
    for i in subject_train:
        produce_data(SavePath, i, subInfo[i-1][0], subInfo[i-1][1], "train")
    for i in subject_val:
        produce_data(SavePath, i, subInfo[i-1][0], subInfo[i-1][1], "val")
    for i in subject_test:
        produce_data(SavePath, i, subInfo[i-1][0], subInfo[i-1][1], "test")
    print(Pos, Neg ,file = ff)

    ff.close()
    f.close()
    ft.close()
    fv.close()
    ftest.close()

    '''
    print("Initializing Datasets and Dataloaders...")
    data_dir = path_MMI
    f = {x: data_dir + x + '.txt' for x in ['train', 'val']}
    image_datasets = {x:MyDataset(root = SavePath , datatxt = x +".txt") for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    phase = "train"
    i = 0
    for img1,img2, labels in dataloaders[phase]:
        i += 1
        if(i >3):break
    '''
