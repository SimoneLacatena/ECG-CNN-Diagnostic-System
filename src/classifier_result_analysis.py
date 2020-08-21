# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np



def load_classifier_result():
    predN_trueA = pd.read_csv('.\\predN_trueA.csv')
    
    predN_trueA = np.asarray(predN_trueA,dtype = int)
    predN_trueA = np.delete(predN_trueA, np.s_[-1:], axis=1)

    
    return predN_trueA
    
    

    
def getAnnotations(track_nro:int,start,end):
    ann = pd.read_csv(annotations + '\\' + str(track_nro) + 'annotations.txt')
    annotaions_values = set()
    annotations_present = list(ann['Y'])
    i = 0
    samples = list(ann['Sample'])
    sample = samples[i]
    while sample < start:
        i += 1
        sample = samples[i]
    
    
    while samples[i] < end:
        annot = annotations_present[i]
        
        annotaions_values.add(annot)
        #print(annotaions_values)
        i += 1
        
        
    return annotaions_values
        
        
        
    
def reports(segments):
    full_report = ' '
    import os
    tracks_names = os.listdir(tracks_dir)
    for tracks in tracks_names:
        tracks = tracks.replace(".csv","")
        tracks_nro = int(tracks[len(tracks) - 3:])
        print(tracks_nro)
        full_report += match(tracks_nro,segments)
        
    return full_report
        
        

    
def match(track_nro:int,segments):
    full_report = ''
    sample_for_segment = len(segments[0])
    
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
                report = ''
                report += 'segment n.ro' + str(i + 1) + ':\n'
                report += '  track n.ro  ' + str(track_nro) + '\n'
                report += '  from sample  ' + str((j)*sample_for_segment) + ' to ' + str((j +1)*sample_for_segment) + '\n'
                report += '  annotations : ' + str(getAnnotations(track_nro,(j)*sample_for_segment,(j +1)*sample_for_segment))+'\n'
                #print(report)
                full_report += report
    return full_report
            
                
    
    
    
tracks_dir =  'D:\\UNIVERSITA\\Università\\Sistemi Multimediali\\sistemiMultimediali-\\Dataset\\mitbih_database\\dataset'   
annotations = 'D:\\UNIVERSITA\\Università\\Sistemi Multimediali\\sistemiMultimediali-\\Dataset\\mitbih_database\\files_preprocessed'
    
    
    
    
    
"""   
def plot(datafilename,segment_index,start=0,end=3,f = 0.00278):
    dataset = pd.read_csv(datafilename)
NA = load_classifier_result()
report = reports(NA)

text_file = open(".\\predictN_trueA.txt", "w")
n = text_file.write(report)
text_file.close()

"""
    
