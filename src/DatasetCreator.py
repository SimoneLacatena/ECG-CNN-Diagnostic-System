# -*- coding: utf-8 -*-


import pandas as pd
import datetime


from matplotlib import pyplot as plt

import numpy





def label(annotations:pd.DataFrame,sample_0, sample_f):
    r,c= annotations.shape
    start = 0
  
    for i in range(r):
        if(annotations.iloc[i,1] >= sample_0):
            start  = i
            break
    while(annotations.iloc[start,1] <= sample_f):
        print(annotations.loc[start])
        if(annotations.iloc[start,2] != 'N'):
            return "A"
                  
        start +=1
    
    return "N"
    


#rilevare i campioni limite ogni 30 secondi
def createDataset(annotation:str,samples:str,dataset_name:str,time_treshold=30,sample_rate = 360):
    annotations = pd.read_csv(annotation)
    data = pd.read_csv(samples)
    limit_samples = []
    
    segment_size = sample_rate*time_treshold
    
    r,c = data.shape
    
    colnames = data.keys()
    segment = dict()
    segments = []
    start = 0
    counter  = 0
    for i in range(r):
         
        segment[colnames[1] + '_' + counter.__str__()] = data.iloc[i,1]
        counter += 1
        
        if(len(segment) >= segment_size):
            print("segment done")
            end = i
            l = label(annotations,start,end)
            print("label done")
            segment['label'] = l
            segments.append(segment)
            segment = dict()
            start = i+1
            counter  = 0
        
    
    print("segment created")
    
            
    #creazione  e memorizazzione del dataset
    dataset = pd.DataFrame(segments)
    print(dataset)
    print("printing dataset")
    dataset.to_csv(dataset_name)
    
    


#dataset = pd.read_csv(filepath + 'Dataset.csv')






def channels(segment_index,dataset):
    ch0= []
    ch1= []
    i = 0
    keys = dataset.keys()
    values = dataset.loc[segment_index]
    while i < len(values) -2:
        if(keys[i] != 'label'):
            ch0.append(values[i])
            ch1.append(values[i+1])
            i+=2
        else :
            i+=1
             
    return ch0,ch1


def plot(datafilename,segment_index,start=0,end=3,f = 0.00278):
    dataset = pd.read_csv(datafilename)
    ch0,ch1 = channels(segment_index,dataset)
    
    x = numpy.arange(0,len(ch0),f)
    x = x[:len(ch0)]
    plt.xlim(start, end)
    plt.ylim(0, 3000)
    
    plt.plot(x,ch0)  
    plt.plot(x,ch1)
    plt.show()
    

    
import os   



def extract_data(filepath,outputDir):
   # filepath  = '.\\Dataset\\mitbih_database\\files_preprocessed'
   # outputDir  = '.\\Dataset\\mitbih_database\\dataset'
    files = os.listdir(filepath)
    
    i = 0
    while i < len(files)-2:
        
        samples = files[i]
        annotations = files[i+1]
        print(samples)
        print(annotations)
        createDataset(filepath +"\\"+annotations,filepath +"\\"+samples,outputDir+"\\data"+samples)
        i+=2
        






    
