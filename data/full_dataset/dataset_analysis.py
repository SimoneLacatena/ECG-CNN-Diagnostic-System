import pandas as  pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
#load dataset

import os
tracks_dir = 'D:\\UNIVERSITA\\Universit\\Sistemi Multimediali\\ECG-CNN-Diagnostic-System\\data\\dataset\\'
annotations = 'D:\\UNIVERSITA\\Universit\\Sistemi Multimediali\\ECG-CNN-Diagnostic-System\\data\\files_preprocessed\\'

def split_train_test():
    dataset = pd.read_csv('end_dataset-15.csv')
    y = dataset['label']
    X = dataset.drop(labels = ['label'],axis  =1)

    xLearning,xTest,yLearning,yTest = train_test_split(X,y,test_size = 0.3)

    trainingset =  pd.concat([xLearning,yLearning], axis=1)
    testset = pd.concat([xTest,yTest], axis=1)


    trainingset.to_csv('train-15.csv',index=False)
    testset.to_csv('test-15.csv',index=False)
    print(trainingset)
    print(testset)

def add_anomalyes(segments,anomalyes):
    tracks_names = os.listdir(tracks_dir)
    print_index  = 1
    
    for tracks in tracks_names:
        print('=' *print_index)
        tracks = tracks.replace(".csv","")
        tracks_nro = int(tracks[len(tracks) - 3:])
        print(tracks_nro)
        anomalyes,matched = match(tracks_nro,segments,anomalyes)
        if matched :
            return anomalyes
        print_index += 1
        
    return anomalyes


def match(track_nro:int,segments,anomalyes):
    sample_for_segment = len(segments[0])
    #print('sfs',sample_for_segment)
    track = pd.read_csv(tracks_dir + '\\data' + str(track_nro) + '.csv')
    track = track.drop(['Unnamed: 0'],axis = 1)
    #print(track)
    track = np.asarray(track)
    track = np.delete(track, np.s_[-1:], axis=1)
    track = np.asarray(track,dtype=int)
    for i in range(len(segments)):
        for j in range(len(track)):
            #print('t',track[j])
            #print('s',segments[i])
            if np.array_equal(track[j], segments[i]):
                print('match')
                anomalyes = getAnnotations(track_nro,(j)*sample_for_segment,(j +1)*sample_for_segment,anomalyes)
                return anomalyes,True
    return anomalyes,False
    
            



def getAnnotations(track_nro:int,start,end,anomalyes):
    print(anomalyes)
    ann = pd.read_csv(annotations + '\\' + str(track_nro) + 'annotations.txt')
    annotations_saw = set()
    annotations_present = list(ann['Y'])
    i = 0
    samples = list(ann['Sample'])
    sample = samples[i]
    while sample < start:
        i += 1
        sample = samples[i]
    
    
    while samples[i] < end:
        annot = annotations_present[i]
        if annot not in annotations_saw and annot != 'N':
            print('anomaly : ', annot)
            try:
                anomalyes[annot] += 1
            except :
                anomalyes[annot] = 1
            
            annotations_saw.add(annot)
        
        #print(annotaions_values)
        i += 1
        
        
    return anomalyes


def analyze(data:pd.DataFrame):
    rows,cols = data.shape
    anomalyes = dict()
    anomalyes['N'] = 0
    data =  np.asarray(data)
    print(data)
    for i in range(rows):
        print('segment n.ro ', i)
        label = data[i,-1]
        if label == 'N':
            anomalyes[label] += 1
            print(anomalyes)
        else :
            example = np.asarray(data[i][:-1],dtype=int)
                
            print(example)
            anomalyes = add_anomalyes([example],anomalyes)
            print(anomalyes)
    
    print(anomalyes)
    file = open(sys.argv[2],'w')
    file.write(str(anomalyes))
    file.close()



dataset  = pd.read_csv(str(sys.argv[1]))
analyze(dataset)




