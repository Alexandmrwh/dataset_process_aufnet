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

INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1

item = [#'01','02',
		# '03','04','05','06','07','08','09',
		# '10','11','12','13','14','15','16','17','18','19',
		# '20','21','22',
		'23','24','25','26','27','28','29',
		'30','31','32']
au_idx = [1, 2, 4, 5, 6, 9 ,12, 17, 25, 26]

logfile = open('./logfile','a')
LeftVideoPath = '../DISFA/Videos_LeftCamera/'
AULabelPath = '../DISFA/ActionUnit_Labels/'
shapePredictorPath = '../DISFA/shape_predictor_68_face_landmarks.dat'

faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)

def get_facelandmark(grayImage):
    global faceDetector, facialLandmarkPredictor
    xyList = []
    left, top = 0, 0
    face = faceDetector(grayImage, 1)
    if len(face) == 0:
        return xyList, left, top

    shape = facialLandmarkPredictor(grayImage, face[0])
    left, top = face[0].left(), face[0].top()
    facialLandmarks = face_utils.shape_to_np(shape)

    for (x, y) in facialLandmarks[0:]:  
        xyList.append(x)
        xyList.append(y)

    return xyList, left, top

def alignment(img, featureList):

    Xs = featureList[::2]
    Ys = featureList[1::2]

    eye_center =((Xs[36] + Xs[45]) * 1./2, (Ys[36] + Ys[45]) * 1./2)
    dx = Xs[45] - Xs[36]
    dy = (Ys[45] - Ys[36])

    angle = math.atan2(dy, dx) * 180. / math.pi
    
    RotationMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0])) 

    return new_img, eye_center

def markAU(FrameLabel,frameIdx,existsAU,au,exists):
	if exists:
		FrameLabel[frameIdx] = FrameLabel[frameIdx] +' 1'
		existsAU[frameIdx] = 1
	else:
		FrameLabel[frameIdx] = FrameLabel[frameIdx] +' 0'

def align_crop_video(print_every=200):
	# for each video
	for idx in item:
		ItemName = 'SN' + str(idx).zfill(3)
		print("Aligning "+ItemName+" ...")

		if not os.path.isfile(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi'):
			print("Failed to find %s"%(ItemName))
			continue

		if not os.path.isdir('../align_disfa/'+ItemName):
			os.mkdir('../align_disfa/'+ItemName)

		# aligning 
		vidLeft = cv2.VideoCapture(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi')
		total_frame = int(vidLeft.get(cv2.CAP_PROP_FRAME_COUNT))
		for t in range(total_frame):
			vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
			isRead, frame = vidLeft.read()
			if isRead:
				# align
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				featureList, _, _ = get_facelandmark(frame_gray)
				if featureList is not None and len(featureList):
					alignimg, _ = alignment(frame, featureList)
					saveImagePath = '../align_disfa/'+ItemName+'/'+ItemName+'_'+str(t)+'.png'
					print(saveImagePath)
					
					cv2.imwrite(saveImagePath, alignimg)
				
			if t % print_every == 0:
				print("aligning: {}".format(t))

		minx, miny, maxx, maxy = INT_MAX, INT_MAX, INT_MIN, INT_MIN
		print("Cropping "+ItemName+" ...")

		if not os.path.isdir('../align_crop_disfa/'+ItemName):
			os.mkdir('../align_crop_disfa/'+ItemName)

		for file in os.listdir('../align_disfa/'+ItemName+'/'):
			frame = cv2.imread('../align_disfa/'+ItemName+'/'+file)
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			face = faceDetector(frame_gray, 1)
			if len(face) == 0:
				print("No face detected")
				continue

			left, top, right, bottom = face[0].left(), face[0].top(), face[0].right(), face[0].bottom()

			# update the largest rect in one session
			if left < minx:
				minx = left
			if top < miny:
				miny = top
			if right > maxx:
				maxx = right
			if bottom > maxy:
				maxy = bottom

		width = maxx - minx
		height = maxy - miny

		saveImagePath = '../align_crop_disfa/'+ItemName+'/'

		for file in os.listdir('../align_disfa/'+ItemName+'/'):
			frame = cv2.imread('../align_disfa/'+ItemName+'/'+file)
			savepath = saveImagePath + file
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


def process(print_every=200):

	# for each video
	for idx in item:
		final_label = open("../DISFA_face_crop_10aus/disfa_10aus_session_{}.txt".format(idx),'w')
		minx, miny, maxx, maxy = INT_MAX, INT_MAX, INT_MIN, INT_MIN
		ItemName = 'SN' + str(idx).zfill(3)
		FrameLabel = [' ']
		existsAU = [0]
		print("Checking "+ItemName+" ...")

		if not os.path.isfile(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi'):
			print("Failed to find %s"%(ItemName))
			continue

		# read video
		vidLeft = cv2.VideoCapture(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi')

		total_frame = 0
		AULabel = AULabelPath + ItemName+ '/' + ItemName +'_au1.txt'
		with open(AULabel,'r') as label:
			total_frame = len(label.readlines())
		for t in range(total_frame):
			FrameLabel.append(' ')
			existsAU.append(0)

		# read every au label txt of the video
		for au in au_idx:
			AULabel = AULabelPath + ItemName+ '/' + ItemName +'_au'+str(au) +'.txt'

			# if the label of that AU doesn't exist
			if not os.path.isfile(AULabel):
				continue

			print("--Checking AU:"+str(au)+" ...")
			with open(AULabel,'r') as label:

				for t,lines in enumerate(label.readlines()):
					frameIdx, AUIntensity = lines.split(',')
					frameIdx, AUIntensity = int(frameIdx),int(AUIntensity)

					if t % print_every ==0:
						print("----Checking Frame:"+str(frameIdx)+" ...")

					markAU(FrameLabel,frameIdx,existsAU,au,exists=(AUIntensity != 0))

		if not os.path.isdir('../DISFA_face_crop_10aus/'+ItemName):
			os.mkdir('../DISFA_face_crop_10aus/'+ItemName)

		for t, label in enumerate(FrameLabel):
			if existsAU[t] == 0:
				continue
			
			vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
			isRead,frame = vidLeft.read()
			if isRead:
				# detect face rect, return in [left, top, right, bottom]
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				face = faceDetector(frame_gray, 1)
				if len(face) == 0:
					print("No face detected in frame{}".format(t))
					continue
				left, top, right, bottom = face[0].left(), face[0].top(), face[0].right(), face[0].bottom()

				# update the largest rect in one session
				if left < minx:
					minx = left
				if top < miny:
					miny = top
				if right > maxx:
					maxx = right
				if bottom > maxy:
					maxy = bottom

			if t % print_every == 0:
				print("face detecting: {}".format(t))
				print("current face rect is: ({}, {}), ({}, {})".format(minx, miny, maxx, maxy))

		print("general face rect is: ({}, {}), ({}, {})".format(minx, miny, maxx, maxy))

		for t,label in enumerate(FrameLabel):
			if existsAU[t]==0:
				continue

			if t % print_every == 0:
				print("Saving frame {}".format(t))

			saveImagePath = '../DISFA_face_crop_10aus/'+ItemName+'/'+ItemName+'_'+str(t)+'.png'
			_path = './'+ItemName+'/'+ItemName+'_'+str(t)+'.png'
			if not os.path.isfile(saveImagePath):
				vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
				isRead,frame = vidLeft.read()
				# crop face and save
				if isRead:
					# cv2.imwrite(saveImagePath,frame[minx: maxx, miny: maxy])
					cv2.imwrite(saveImagePath,frame[miny: maxy, minx: maxx])

					print("Saved",saveImagePath,file=logfile)
			else:
				print(saveImagePath, "exists.",file=logfile)
			print(_path,label,file=final_label)
			

		final_label.close()


if __name__ == '__main__':
	# process()
	align_crop_video()

