import numpy as np
import pickle
import os
import csv
au_idx = [1, 2, 4, 5, 6, 9 ,12, 15, 17, 25, 26]
au_nums = len(au_idx)
DisfaAllLabelPath = "../data/DISFA/DISFA_11_label/"
DisfaImagePath = "../data/DISFA/"

stride = 1
samplingrate = 10

def split4flow():
    data = open(DisfaAllLabelPath+'finetuneflow'+'.txt','w')
    
    # for each subject
    for idx in range(1, 33):
        txt_name = str(idx).zfill(2)
        session_label_path = DisfaAllLabelPath + "disfa_11aus_session_{}.txt".format(txt_name)
        # for each session
        if os.path.isfile(session_label_path):
            session_label = open(session_label_path,'r')
            all_samples = session_label.readlines()
            sample_num = len(all_samples)
            for i in range(0, sample_num - stride * samplingrate, stride * samplingrate):
                line1 = all_samples[i]
                frameIdx1 = line1.split()[0]
                # get new path for frameidx
                frameIdx1 = frameIdx1[2: ]
                frameIdx1 = DisfaImagePath + frameIdx1
                line2 = all_samples[i + stride * samplingrate - 1]
                frameIdx2 = line2.split()[0]
                frameIdx2 = frameIdx2[2: ]
                frameIdx2 = DisfaImagePath + frameIdx2
                print(frameIdx1, frameIdx2, file=data)
    data.close()

if __name__ == "__main__":
    split4flow()