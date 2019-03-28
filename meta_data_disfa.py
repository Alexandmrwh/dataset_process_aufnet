import numpy as np
import os
au_idx = [1, 2, 4, 5, 6, 9 ,12, 17, 25, 26]

def get_meta_data():
	total_meta_data = open("../DISFA_10aus/DISFA_meta_data.txt",'w')
	print(au_idx, file=total_meta_data)
	for idx in range(1, 33):
		au_pos_sum = np.zeros(10)
		total_frame = 0
		txt_name = str(idx).zfill(2)
		session_label_path = "../DISFA_face_crop_10aus/disfa_10aus_session_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path,'r')
			for t, lines in enumerate(session_label.readlines()):
				total_frame += 1
				au_pos_tmp = np.zeros(10)
				frameIdx, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9]= lines.split()
				for i in range(10):
					au_pos_sum[i] += int(au_pos_tmp[i])

			print("session {}:".format(idx), au_pos_sum, "total_frame: {}".format(total_frame), file=total_meta_data)

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
			for t, lines in enumerate(session_label.readlines()):
				print(lines, file=all_labels)
	all_labels.close()





if __name__ == '__main__':
	# get_meta_data()
	get_all_labels()
			
			



