import numpy as np
import os
path_DISFA = '../DISFA_10aus/DISFA_face_crop_10aus/'
path_result = '../DISFA_10aus/'
stride = 2
au_num = 10
train_idx = ['1']
val_idx=['2']
test_idx=['3']

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
        session_label_path = path_DISFA + "disfa_10aus_session_{}.txt".format(txt_name)
        print(session_label_path)
        if os.path.isfile(session_label_path):
            session_label = open(session_label_path,'r')
            all_samples = session_label.readlines()
            sample_num = len(all_samples)
            aus = np.zeros(10)
            for i in range(0, sample_num-stride):
                line1 = all_samples[i]
                frameIdx1, aus[0], aus[1], aus[2], aus[3], \
                           aus[4], aus[5], aus[6], aus[7], aus[8], aus[9]= line1.split()
                aus = aus.astype(np.int)
                line2 = all_samples[i+stride]
                frameIdx2, _, _, _, _, \
                           _, _, _, _, _, _, = line2.split()
                print(frameIdx1, frameIdx2, aus[0], aus[1], aus[2], aus[3], \
                           aus[4], aus[5], aus[6], aus[7], aus[8], aus[9], file=data)
    data.close()

if __name__ == '__main__':
    train = 'train'
    val = 'val'
    test = 'test'
    split(train)
    split(val)
    split(test)




    
