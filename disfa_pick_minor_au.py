import numpy as np
import os
au_idx = [1, 2, 4, 5, 6, 9 ,12, 17, 25, 26]
minor_au = [1, 2, 9, 17, 25]
minor_au_idx = [0, 1, 5, 7, 8]

def pick_minor_au():
	for idx in range(1, 33):
		txt_name = str(idx).zfill(2)
		session_label_path = "../DISFA_10aus/DISFA_face_crop_10aus/DISFA_all_label/disfa_10aus_session_{}.txt".format(txt_name)
		picked_label_path = "../DISFA_10aus/DISFA_face_crop_10aus/DISFA_picked_label/disfa_10aus_picked_{}.txt".format(txt_name)

		# read the label file of each session
		if os.path.isfile(session_label_path):
			session_label = open(session_label_path,'r')
			picked_data = open(picked_label_path, 'w')
			for t, lines in enumerate(session_label.readlines()):
				lines = lines.strip('\n')
				au_pos_tmp = np.zeros(10)
				frameIdx, au_pos_tmp[0], au_pos_tmp[1], \
					au_pos_tmp[2], au_pos_tmp[3], \
					au_pos_tmp[4], au_pos_tmp[5], au_pos_tmp[6],\
				 	au_pos_tmp[7], au_pos_tmp[8], au_pos_tmp[9]= lines.split()
				for i in range(len(minor_au)):
					if int(au_pos_tmp[minor_au_idx[i]]) == 1:
						print(lines, file=picked_data)
						break

if __name__ == '__main__':
	pick_minor_au()
			





