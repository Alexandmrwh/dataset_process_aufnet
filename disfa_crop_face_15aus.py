#!/usr/bin/env python
#coding=utf-8

'''
This script is to split frames into images 
'''
import os
import cv2
import csv
import dlib
import sys

INT_MAX = sys.maxsize  

INT_MIN = -sys.maxsize-1

item = [#'01'
		'02','03','04','05','06','07','08','09',
		'10','11','12','13','14','15','16','17','18','19',
		'20','21','22','23','24','25','26','27','28','29',
		'30','31','32']
au_idx = [1, 2, 4, 5, 6, 7, 9 ,10 ,12, 17 ,23 ,24, 25 ,26 ,43]

logfile = open('./logfile','a')
LeftVideoPath = '../DISFA/Videos_LeftCamera/'
AULabelPath = '../DISFA/ActionUnit_Labels/'

def markAU(FrameLabel,frameIdx,existsAU,au,exists):
	if exists:
		FrameLabel[frameIdx] = FrameLabel[frameIdx] +' 1'
		existsAU[frameIdx] = 1
	else:
		FrameLabel[frameIdx] = FrameLabel[frameIdx] +' 0'
	

def process(print_every=200):

	# for each video
	for idx in item:
		final_label = open("../DISFA_face_crop_15aus/disfa_15aus_session_{}.txt".format(idx),'w')
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

		if not os.path.isdir('../DISFA_face_crop_15aus/'+ItemName):
			os.mkdir('../DISFA_face_crop_15aus/'+ItemName)

		for t, label in enumerate(FrameLabel):
			if existsAU[t] == 0:
				continue
			
			vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
			isRead,frame = vidLeft.read()
			if isRead:
				# detect face rect, return in [left, top, right, bottom]
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faceDetector = dlib.get_frontal_face_detector()
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

			saveImagePath = '../DISFA_face_crop_15aus/'+ItemName+'/'+ItemName+'_'+str(t)+'.png'
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
	process()

