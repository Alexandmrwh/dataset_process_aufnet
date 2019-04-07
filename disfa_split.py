import numpy as np
import os
path_DISFA = '../DISFA_10aus/DISFA_face_crop_10aus/'
path_DISFA_picked_label = '../DISFA_10aus/DISFA_face_crop_10aus/DISFA_picked_label/'
path_result = '../DISFA_10aus_missingrate0.25/'
stride = 4 # missing rate = 0.25
au_num = 10
train_idx = [2,3,4,6,7,8,12,13,17,18,23,26,28,29,30,32]
val_idx=[1,5,9,10,11,14,15,16,19,20,21,22,24,25,27,31]


def split(type):
    data = open(path_result+type+'.txt','w')
    if type == 'train':
        type_idx = train_idx
    elif type == 'val':
        type_idx = val_idx
    elif type == 'test':
        type_idx = test_idx

    for session_idx in type_idx:
        txt_name = str(session_idx).zfill(2)
        session_label_path = path_DISFA_picked_label + "disfa_10aus_picked_{}.txt".format(txt_name)
        print(session_label_path)
        if os.path.isfile(session_label_path):
            session_label = open(session_label_path,'r')
            all_samples = session_label.readlines()
            sample_num = len(all_samples)
            aus = np.zeros(10)
            for i in range(0, sample_num-stride+1, stride):
                line1 = all_samples[i]
                frameIdx1, aus[0], aus[1], aus[2], aus[3], \
                           aus[4], aus[5], aus[6], aus[7], aus[8], aus[9]= line1.split()
                for j in range(1, stride):
                aus = aus.astype(np.int)
                line2 = all_samples[i+j]
                frameIdx2, _, _, _, _, \
                        _, _, _, _, _, _, = line2.split()
                print(frameIdx1, frameIdx2, aus[0], aus[1], aus[2], aus[3], \
                        aus[4], aus[5], aus[6], aus[7], aus[8], aus[9], file=data)
    data.close()

if __name__ == '__main__':
    train = 'train'
    val = 'val'
    # test = 'test'
    split(train)
    split(val)
    # split(test)




    
