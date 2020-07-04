# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os

directory =  '.\\Dataset\\mitbih_database'

for file in os.listdir(directory):
    if file.endswith("annotations.txt"):
        try:
            print(file)
            data=pd.read_csv(directory + "\\"+ file ,delim_whitespace=True)
            
            print(data)
            data = data.drop(["Type", "Sub", "Chan","Num","Aux"], axis=1)
          
            
            data.rename(columns={"#": "Y"},inplace=True)
            r,c=data.shape
            for i in range(r):
                if(data.iloc[i,2]=="+"):
                    data.drop(data.index[i])
                
            
            data.to_csv(directory + "\\files_preprocessed\\" + file,index=False)
        except:
            continue

