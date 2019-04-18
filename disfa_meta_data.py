import numpy as np
import os
au_idx = [1, 2, 4, 5, 6, 9 ,12, 17, 25, 26]

def get_meta_data():
	total_meta_data = open("../DISFA_10aus/DISFA_meta_data.txt",'w')
	print(au_idx, file=total_meta_data)
	sum = np.zeros(10)
	for idx in range(1, 33):
		au_pos_sum = np.zeros(10)
		total_frame = 0
		txt_name = str(idx).zfill(2)
		session_label_path = "../DISFA_10aus/DISFA_face_crop_10aus/DISFA_all_label/disfa_10aus_session_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path,'r')
			for _, lines in enumerate(session_label.readlines()):
				total_frame += 1
				au_pos_tmp = np.zeros(10)
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9]= lines.split()
				for i in range(10):
					au_pos_sum[i] += int(au_pos_tmp[i])

			print("session {}:".format(idx), au_pos_sum, "total_frame: {}".format(total_frame), file=total_meta_data)
			for i in range(10):
				sum[i] += int(au_pos_sum[i])
	print("total:", sum, file=total_meta_data)
	total_meta_data.close()

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
	sum = np.zeros(10)
	for idx in range(1, 33):
		au_pos_sum = np.zeros(10)
		total_frame = 0
		txt_name = str(idx).zfill(2)
		session_label_path = "../DISFA_10aus/DISFA_face_crop_10aus/DISFA_picked_label/disfa_10aus_picked_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path, 'r')
			for _, lines in enumerate(session_label.readlines()):
				total_frame += 1
				au_pos_tmp = np.zeros(10)
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9]= lines.split()
				for i in range(10):
					au_pos_sum[i] += int(au_pos_tmp[i])
			for i in range(10):
				sum[i] += int(au_pos_sum[i])

			print("session {}:".format(idx), au_pos_sum, "total_frame: {}".format(total_frame), file=picked_meta_data)
	print("total:", sum, file=picked_meta_data)

	picked_meta_data.close()

def get_prob_distribution():
	prob_distribution = open("../data/DISFA/DISFA_prob_distribution.txt", 'w')
	single_occurance = np.zeros(len(au_idx))
	co_occurance = np.zeros([len(au_idx), len(au_idx)])
	print(au_idx, file=prob_distribution)
	
	# for each session
	for idx in range(1, 33):
		txt_name = str(idx).zfill(2)
		session_label_path = "../data/DISFA/DISFA_all_label/disfa_10aus_session_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path, 'r')
			for _, lines in enumerate(session_label.readlines()):
				au_pos_tmp = np.zeros(10)
				_, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9]= lines.split()
				for i in range(len(au_idx)):
					single_occurance[i] += int(au_pos_tmp[i])
				for i in range(len(au_idx)):
					for j in range(i+1, len(au_idx)):
						if au_pos_tmp[i] and au_pos_tmp[j]:
							co_occurance[i][j] += 1
							co_occurance[j][i] += 1
	print(single_occurance.astype(int), file=prob_distribution)
	print(co_occurance.astype(int), file=prob_distribution)

if __name__ == '__main__':
	# get_meta_data()
	# get_all_labels()
	# get_picked_meta_data()
	get_prob_distribution()
			
			




