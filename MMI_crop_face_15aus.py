import sys
import cv2
import os
import dlib
import numpy as np
from PIL import Image
import xml.dom.minidom

subInfo = [[1,120],[119,245],[245,351],[351,471],[471,580],[580,690],[689,795],[793,860],[859,965],[964,1067],
[1067,1142],[1142,1215],[1215,1287],[1287,1416],[1416,1500],[1500,1569]]

left_right = [2,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0] # 1 represent the face is in the right half picture 0 left

# subject_train = [1,2,3,4,5,6,9,14] 
# subject_val = [7,8,15,16]
# subject_test = [10,11,12,13]

subject_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

AU = [1,2,4,5,6,7,9,10,12,17,23,24,25,26,43]
Pos = np.zeros([17,len(AU)])
Neg = np.zeros([17,len(AU)])
path_MMI = '../MMI_crop_face_15aus/'

def _locate_faces(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceDetector = dlib.get_frontal_face_detector()
	face = faceDetector(frame_gray, 1)
	if len(face) == 0:
		print("No face detected in frame{}".format(t))
		continue
	left, top, right, bottom = face[0].left(), face[0].top(), face[0].right(), face[0].bottom()
	return left, top, right, bottom

def produce_data(SavePath, SubNum, subDown, subUp, status):
    print ("Subject: ", SubNum)
    ReadPath = "/home/ubuntu/AU-WorkSite/FAC_DataBase/MMIdatabase/subject"+str(SubNum)+"/Sessions/"
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
        ty = []
        ActionUnit = root.getElementsByTagName('ActionUnit')
        for au in ActionUnit:
            ty.append(au.getAttribute('Number')) # ty contains the au numbers that occured in the avi
        #print(ty)
        for index in range(len(AU)):
            if str(AU[index]) in ty: # 4 here means au4
                print ("Session", SessionNum, "labeled with au",AU[index])
            else:
                continue
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

            vc = cv2.VideoCapture(VideoPath)
            read_value, webcam_image = vc.read()
            cur_frame = 0
            i = 1

            image = Image.fromarray(webcam_image)
            width, height = image.size[:2]
            box_1 = (0,0,width/2,height)
            box_2 = (width/2,0,width,height)
            SavePath4 = ''

            while read_value:
                X = []
                cur_frame += 1  

                if(cur_frame == FrameNum/3):

                    SavePath4 = SavePath+"sub"+str(SubNum)+"Session"+str(SessionNum)+'_'+str(i)+".png"

                if(cur_frame > FrameNum/3 and cur_frame < FrameNum *2 /3):

                    SavePath5 = SavePath+"sub"+str(SubNum)+"Session"+str(SessionNum)+'_'+str(i)+".png"

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

if (__name__ == '__main__'):
    ff = open(path_MMI + 'pos_neg.txt','w')
    f = open(path_MMI+'out.txt','w')
    ft = open(path_MMI + 'train.txt','w')
    fv = open(path_MMI + 'val.txt','w')
    ftest = open(path_MMI+'test.txt','w')

    
    for i in subject_idx:
        produce_data(path_MMI, i, subInfo[i-1][0], subInfo[i-1][1], "MMI_15aus")

    print(Pos, Neg ,file = ff)

    ff.close()
    f.close()
    ft.close()
    fv.close()
    ftest.close()