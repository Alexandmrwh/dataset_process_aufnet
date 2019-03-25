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

item = ['01','02','03','04','05','06','07','08','09',
		'10','11','12','13','14','15','16','17','18','19',
		'20','21','22','23','24','25','26','27','28','29',
		'30','31','32']
au_idx = [1, 2, 4, 5, 6, 7, 9 ,10 ,12, 17 ,23 ,24, 25 ,26 ,43]

au2arrayidx = {'1':0,
				'2':1,
				'4':2,
				'12':3,
				'25':4,
				'26':5}

logfile = open('./logfile','a')
LeftVideoPath = './Videos_LeftCamera/'
AULabelPath = './ActionUnit_Labels/'
savePath = './'

def markAU(FrameLabel,frameIdx,existsAU,au,exists):
	if exists:
		FrameLabel[frameIdx] = FrameLabel[frameIdx] +' 1'
		existsAU[frameIdx] = 1
	else:
		FrameLabel[frameIdx] = FrameLabel[frameIdx] +' 0'
	

def process(print_every=200):
	final_label = open("disfa_15aus_label_0325.txt",'w')

	# for each video
	for idx in item:
		minx, miny, maxx, maxy = sys.maxint, sys.maxint, sys.minint, sys.minint
		ItemName = 'SN' + str(idx).zfill(3)
		FrameLabel = [' ']
		existsAU = [0]
		print("Checking "+ItemName+" ...")

		if not os.path.isfile(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi'):
			print("Failed to find %s"%(ItemName))
			continue

		# read video
		vidLeft = cv2.VideoCapture(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi')

		# read every au label txt of the video
		for au in au_idx:
			AULabel = AULabelPath + ItemName+ '/' + ItemName +'_au'+str(au) +'.txt'

			# if the label of that AU doesn't exist
			if not os.path.isfile(AULabel):
				continue

			print("--Checking AU:"+str(au)+" ...")
			# for one au, 
			with open(AULabel,'r') as label:

				for t,lines in enumerate(label.readlines()):
					frameIdx, AUIntensity = lines.split(',')
					frameIdx, AUIntensity = int(frameIdx),int(AUIntensity)

					if t % print_every ==0:
						print("----Checking Frame:"+str(frameIdx)+" ...")

					markAU(FrameLabel,frameIdx,existsAU,au,exists=(AUIntensity != 0))

		if not os.path.isdir('./DISFA_in_au/'+ItemName):
			os.mkdir('./DISFA_in_au/'+ItemName)

		for t, label in enumerate(FrameLabel):
			if existsAU[t] == 0:
				continue
			if t % print_every == 0:
				print(t, existsAU[t])
				print("face detecting: {}".format(t))

			saveImagePath = './DISFA_in_au/'+ItemName+'/'+ItemName+'_l_'+str(t)+'.png'
			if not os.path.isfile(saveImagePath):
				vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
				isRead,frame = vidLeft.read()

				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faceDetector = dlib.get_frontal_face_detector()
				face = faceDetector(frame_gray, 1)
				if len(face) == 0:
					print("No face detected in {}".format(saveImagePath))
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
		print("general face rect is:({}, {}), ({}, {})".format(minx, miny, maxx, maxy))

		for t,label in enumerate(FrameLabel):
			if existsAU[t]==0:
				continue

			if t % print_every == 0:
				print(t,existsAU[t])
				print("Saving frame %d"%(t))

			saveImagePath = './DISFA_in_au/'+ItemName+'/'+ItemName+'_l_'+str(t)+'.png'
			_path = './'+ItemName+'/'+ItemName+'_l_'+str(t)+'.png'
			if not os.path.isfile(saveImagePath):
				vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
				isRead,frame = vidLeft.read()

				cv2.imwrite(saveImagePath,frame[minx: maxx, miny: maxy])
				print("Saved",saveImagePath,file=logfile)
			else:
				print(saveImagePath, "exists.",file=logfile)
			print(_path,label,file=final_label)
			

	final_label.close()

def avi2png(print_every=100):
	'''
	Transform avi to png and save them in  path './Videos_[(Left)(Right)]Camera/frame/SN0**'
	'''
	for idx in item:
		vidLeft = cv2.VideoCapture(LeftVideoPath+'LeftVideoSN0'+idx+'_comp.avi')

		ItemName = 'SN' + str(idx).zfill(3)
		print("Checking "+ItemName+" ...")

		if not os.path.isdir('./DISFA_in_au/'+ItemName):
			continue

		FrameLabel = open(AULabelPath+ItemName+'/'+ItemName+'_au1.txt','r')
		for line in FrameLabel.readlines():
			t,line = line.split(',')
			t = int(t)

			
			if t % print_every == 0:
				print("Saving frame %d"%(t))

			saveImagePath = './DISFA_in_au/'+ItemName+'/'+ItemName+'_l_'+str(t)+'.png'
			if not os.path.isfile(saveImagePath):
				vidLeft.set(cv2.CAP_PROP_POS_FRAMES,t)
				isRead,frame = vidLeft.read()
				cv2.imwrite(saveImagePath,frame)
				print("Saved",saveImagePath,file=logfile)
			else:
				print(saveImagePath, "exists.",file=logfile)
			
			saveImagePath = './DISFA_in_au/'+ItemName+'/'+ItemName+'_r_'+str(t)+'.png'
			if os.path.isfile('./DISFA_in_au/'+ItemName+'/'+'r_'+str(t)+'.png'):
				os.rename('./DISFA_in_au/'+ItemName+'/'+'r_'+str(t)+'.png',saveImagePath)
				print("Changed name from",'./DISFA_in_au/'+ItemName+'/'+'r_'+str(t)+'.png to ',saveImagePath,file=logfile)
			elif not os.path.isfile(saveImagePath):
				vidRight.set(cv2.CAP_PROP_POS_FRAMES,t)
				isRead,frame = vidRight.read()
				cv2.imwrite(saveImagePath,frame)
				print("Saved",saveImagePath,file=logfile)
			else:
				print(saveImagePath, "exists.",file=logfile)


if __name__ == '__main__':
	process()

