# -*- coding: utf-8 -*-


import pandas as pd

from matplotlib import pyplot as plt

import numpy

import os   





"""
funzione label

input: 
    annotations : Dataframe contente le annotazioni
    sample_0 : numero del campione iniziale del segmento
    sample_f :numero del campione finale del segmento

descrizione:
    la funzione controlla i vari picchi presenti nel segmento utilizzando annotations
    e se rileva un anomalia resituisce la label A , se invece non rileva alcuna anomalia restiuisce 
    la label N

"""
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
    

"""
funzione createDataset

input: 
    annotation : filepath del file contentente le annotazioni
    samples : filepath del csv contentente i campioni del tracciato
    dataset_name : filepath del file csv in cui espotare il dataset creato
    time_treshold:  numero di secondi per segmento
    sample_rate : frequenza di campionamento con cui è satato campionato i segnale

descrizione:
    la funzione frammenta il file samples in segmento di time_treshold secondi e gli assegna un etichetta
"""
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
    
    

"""
def plot(datafilename,segment_index,start=0,end=3,f = 0.00278):
    dataset = pd.read_csv(datafilename)
    y = list(dataset.loc[segment_index])
    x = numpy.arange(0,len(y),f)
    x = x[:len(y)]
    plt.xlim(start, end)
    plt.ylim(0, 3000)
    plt.plot(x,y)  
    plt.show()
    

""" 



"""
funzione extract_data

input: 
    filepart : filepath della cartella dove si trovano i files contenti i campioni e le annotazioni
    outputDir : filepath del cartalle in cui memorizzare i csv

descrizione:
    la funzione processa i dati presenti il filepath e resituisce dei csv uno per ogni coppia di file 
    es. 100,100annotations  -> data100

"""

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
        
"""
funzione union

input: 
    directorypath : filepath della cartella dove si trovano i csv da unire
    utputdir: fielpath della cortella che conterra l'unic csv'

descrizione:
    la funzione unisce tutti csv presenti in un unico csv

"""
def union(directorypath:str,outputdir:str):
    data = os.listdir(directorypath)
    dataset =  pd.read_csv(directorypath +"\\" + data[0],index_col=0)
    for i in range(1,len(data)):
        print(data[i])
        data_i  = pd.read_csv(directorypath +"\\" + data[i],index_col=0)
        dataset = pd.concat([dataset, data_i], ignore_index=True)
        
    
    dataset.to_csv(outputdir + "\\end_dataset.csv",index=False)
        
        
    





    
