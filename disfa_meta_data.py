import numpy as np
import pickle
import os
import csv
au_idx = [1, 2, 4, 5, 6, 9 ,12, 15, 17, 25, 26]
au_nums = len(au_idx)
DisfaAllLabelPath = "../data/DISFA/DISFA_11_label/"

def get_meta_data():
	total_meta_data = open(DisfaAllLabelPath + "DISFA_meta_data.txt",'w')
	print(au_idx, file=total_meta_data)
	sum = np.zeros(au_nums)
	sumcsv = np.zeros(shape = [27, au_nums+1], dtype=int)
	sumidx = 0
	for idx in range(1, 33):
		au_pos_sum = np.zeros(au_nums)
		total_frame = 0
		txt_name = str(idx).zfill(2)
		session_label_path = DisfaAllLabelPath + "disfa_11aus_session_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path,'r')
			for _, lines in enumerate(session_label.readlines()):
				total_frame += 1
				au_pos_tmp = np.zeros(au_nums)
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9], au_pos_tmp[10]= lines.split()
				for i in range(au_nums):
					au_pos_sum[i] += int(au_pos_tmp[i])
				
			print("session {}:".format(idx), au_pos_sum, "total_frame: {}".format(total_frame), file=total_meta_data)
			for i in range(au_nums):
				sum[i] += int(au_pos_sum[i])
			sumcsv[sumidx][0] = idx
			sumcsv[sumidx][1:] = au_pos_sum
			sumidx += 1
	print("total:", sum, file=total_meta_data)
	total_meta_data.close()

	sumcsvfile = open(DisfaAllLabelPath+'DISFA_meta.csv', 'w')
	writer = csv.writer(sumcsvfile, delimiter = ',')
	for line in range(sumcsv.shape[0]):
		writer.writerow(sumcsv[line])
	sumcsvfile.close()

def get_all_labels():
	all_labels = open("../DISFA_10aus/all_labels.txt", 'w')
	for idx in range(1, 33):
		print("writing {}...".format(idx))
		txt_name = str(idx).zfill(2)
		session_label_path = "../DISFA_10aus/DISFA_face_crop_10aus/disfa_10aus_session_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path,'r')
			for _, lines in enumerate(session_label.readlines()):
				print(lines, file=all_labels)
	all_labels.close()

def get_picked_meta_data():
	picked_meta_data = open("../DISFA_10aus/DISFA_picked_meta_data.txt",'w')
	print(au_idx, file=picked_meta_data)
	sum = np.zeros(au_nums)
	for idx in range(1, 33):
		au_pos_sum = np.zeros(au_nums)
		total_frame = 0
		txt_name = str(idx).zfill(2)
		session_label_path = "../DISFA_10aus/DISFA_face_crop_10aus/DISFA_picked_label/disfa_10aus_picked_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path, 'r')
			for _, lines in enumerate(session_label.readlines()):
				total_frame += 1
				au_pos_tmp = np.zeros(au_nums)
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9], au_pos_tmp[10]= lines.split()
				for i in range(au_nums):
					au_pos_sum[i] += int(au_pos_tmp[i])
			for i in range(au_nums):
				sum[i] += int(au_pos_sum[i])

			print("session {}:".format(idx), au_pos_sum, "total_frame: {}".format(total_frame), file=picked_meta_data)
	print("total:", sum, file=picked_meta_data)

	picked_meta_data.close()

def get_prob_distribution():
	prob_distribution = open("../data/DISFA/DISFA_prob_distribution.txt", 'w')
	prob_distribution_pkl = open("../data/DISFA/DISFA_prob_distribution.pkl", 'wb')
	single_occurance = np.zeros(au_nums)
	co_occurance = np.zeros([au_nums, au_nums])
	print(au_idx, file=prob_distribution)
	
	# for each session
	for idx in range(1, 33):
		txt_name = str(idx).zfill(2)
		session_label_path = "../data/DISFA/DISFA_all_label/disfa_10aus_session_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path, 'r')
			for _, lines in enumerate(session_label.readlines()):
				au_pos_tmp = np.zeros(au_nums)
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9], au_pos_tmp[10]= lines.split()
				for i in range(au_nums):
					single_occurance[i] += int(au_pos_tmp[i])
				for i in range(au_nums):
					for j in range(i+1, au_nums):
						if au_pos_tmp[i] and au_pos_tmp[j]:
							co_occurance[i][j] += 1
							co_occurance[j][i] += 1
	print(single_occurance.astype(int), file=prob_distribution)
	print(co_occurance.astype(int), file=prob_distribution)

	pkl_dict = {'nums': single_occurance.astype(int),
				'adj': co_occurance.astype(int)}
	pickle.dump(pkl_dict, prob_distribution_pkl)

	prob_distribution.close()
	prob_distribution_pkl.close()


if __name__ == '__main__':
	get_meta_data()
	# get_all_labels()
	# get_picked_meta_data()
	# get_prob_distribution()
			
			




