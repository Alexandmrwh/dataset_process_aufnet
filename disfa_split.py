path_DISFA = './'
au_number = 10
au_idx = ['1','2','4','5','6','9','12','17','24','25']

def split():
    train_idx = ['SN001']
    val_idx=['SN002']
    test_idx=['SN003']


    label = open(path_DISFA+'all_labels.txt','r')

    total_data = open(path_DISFA+'total.txt','w')
    test_data = open(path_DISFA+'test.txt','w')
    val_data = open(path_DISFA+'val.txt','w')
    train_data = open(path_DISFA+'train.txt','w')
    lines = []

    ## the first line
    tmp = label.readline()
    lines.append(tmp)
    line = lines[0]
    line = line.strip('\n')
    line = line.rstrip()
    words = line.split()

    img_path = words[0]
    item_name = img_path.split('/')[-2]
    img_name = img_path.split('/')[-1]
    au_label = ''
    for au in range(au_number):
        au_label = au_label+' '+words[au+1]
    au_label_list = [words[au] for au in range(1,1+au_number)]

    lines[0] = [item_name,img_name,au_label,au_label_list]

    # the loop
    while True:
        tmp = label.readline()

        # EOF
        if tmp==None or tmp == '':
            label.close()
            total_data.close()
            train_data.close()
            val_data.close()
            test_data.close()
            return
            
        lines.append(tmp)
        print('_______________________')
        print(lines)
        for i in range(1,2):
            line = lines[i]
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()

            img_path = words[0]
            item_name = img_path.split('/')[-2]
            img_name = img_path.split('/')[-1]
            au_label = ''
            for au in range(au_number):
                au_label = au_label+' '+words[au+1]
            au_label_list = [words[au] for au in range(1,7)]

            lines[i] = [item_name,img_name,au_label,au_label_list]

        # if from different items, abort this
        if lines[0][0]!=lines[1][0]:
            lines.remove(lines[0])
            continue

        
        readPath = './'+item_name+'/'+img_name
        img = cv2.imread(readPath)
        try:
            w,h,c = img.shape
            cropped=img[0:w,int((h-w)/2):int((h+w)/2)]
        except:
            lines.remove(lines[0])
            continue


        if lines[0][0] in test_idx:
            print('./test/'+lines[0][1],\
                        './test/'+lines[1][1],\
                        lines[0][2],file=test_data)
            if os.path.isfile('./test/'+lines[0][1]):
                lines.remove(lines[0])
                continue;
            cv2.imwrite('./test/'+lines[0][1],cropped)

        elif lines[0][0] in val_idx:
            print('./val/'+lines[0][1],\
                        './val/'+lines[1][1],\
                        lines[0][2],file=val_data)
            if os.path.isfile('./val/'+lines[0][1]):
                lines.remove(lines[0])
                continue;
            cv2.imwrite('./val/'+lines[0][1],cropped)

        else:

            print('./train/'+lines[0][1],\
                        './train/'+lines[1][1],\
                        lines[0][2],file=train_data)
            if os.path.isfile('./train/'+lines[0][1]):
                lines.remove(lines[0])
                continue;
            cv2.imwrite('./train/'+lines[0][1],cropped)

        print('./total/'+lines[0][1],\
                        './total/'+lines[1][1],\
                        lines[0][2],file=total_data)
        print("remove!!!",lines[0])
        lines.remove(lines[0])
