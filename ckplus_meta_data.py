#!/usr/bin/env python
#coding=utf-8

import os
import sys
import pickle
import numpy as np

CKPlusPickedLabels = '../data/Cohn-Kanade/CK+/CK+_Picked_label/'
INT_MAX = sys.maxsize  
INT_MIN = -sys.maxsize-1

au_idx_lair = [1, 2, 4, 5, 6, 
				7, 9, 10, 12, 17, 
				23, 24, 25, 26, 43]

def get_meta_data():
	total_meta_data = open(CKPlusPickedLabels+'CKPlus_meta_data.txt', 'w')
	print(au_idx_lair, file=total_meta_data)
	sum = np.zeros(len(au_idx_lair))
	for subidx in range(1000):
		# for each subject
		total_session = 0
		au_pos_sum = np.zeros(len(au_idx_lair))

		sublabelpath = CKPlusPickedLabels + 'S' + str(subidx).zfill(3) + '.txt'
		if os.path.isfile(sublabelpath):
			sublabel = open(sublabelpath, 'r')
			for _, lines in enumerate(sublabel.readlines()):
				total_session += 1
				au_pos_tmp = np.zeros(len(au_idx_lair))
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9], \
				 	au_pos_tmp[10], au_pos_tmp[11], au_pos_tmp[12],\
				 	au_pos_tmp[13], au_pos_tmp[14] = lines.split()
				for i in range(len(au_idx_lair)):
					au_pos_sum[i] += int(au_pos_tmp[i])
			for i in range(len(au_idx_lair)):
				sum[i] += au_pos_sum[i]

			print("subject {}:".format(subidx), au_pos_sum, "total_session: {}".format(total_session), file=total_meta_data)
	print("total:", sum, file=total_meta_data)

	total_meta_data.close()

def get_prob_distribution():
	prob_distribution = open(CKPlusPickedLabels+'CKPlus_prob_distribution.txt', 'w')
	prob_distribution_pkl = open(CKPlusPickedLabels+'CKPlus_prob_distribution.pkl', 'wb')

	single_occurance = np.zeros(len(au_idx_lair))
	co_occurance = np.zeros([len(au_idx_lair), len(au_idx_lair)])
	print(au_idx_lair, file=prob_distribution)

	for subidx in range(1000):
		# for each subject
		sublabelpath = CKPlusPickedLabels + 'S' + str(subidx).zfill(3) + '.txt'
		if os.path.isfile(sublabelpath):
			sublabel = open(sublabelpath, 'r')
			for _, lines in enumerate(sublabel.readlines()):
				au_pos_tmp = np.zeros(len(au_idx_lair))
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9], \
				 	au_pos_tmp[10], au_pos_tmp[11], au_pos_tmp[12],\
				 	au_pos_tmp[13], au_pos_tmp[14] = lines.split()
				for i in range(len(au_idx_lair)):
					single_occurance[i] += int(au_pos_tmp[i])
				for i in range(len(au_idx_lair)):
					for j in range(i+1, len(au_idx_lair)):
						if au_pos_tmp[i] and au_pos_tmp[j]:
							co_occurance[i][j] += 1
							co_occurance[j][i] += 1
	print(single_occurance.astype(int), file = prob_distribution)
	print(co_occurance.astype(int), file=prob_distribution)

	pkl_dict = {'nums': single_occurance.astype(int),
				'adj': co_occurance.astype(int)}
	pickle.dump(pkl_dict, prob_distribution_pkl)

	prob_distribution.close()
	prob_distribution_pkl.close()


if __name__ == '__main__':
	# get_meta_data()
	get_prob_distribution()