import pandas as  pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
#load dataset
def split_train_test():
    dataset = pd.read_csv(sys.argv[1])
    y = dataset['label']
    X = dataset.drop(labels = ['label'],axis  =1)

    xLearning,xTest,yLearning,yTest = train_test_split(X,y,test_size = 0.3)

    trainingset =  pd.concat([xLearning,yLearning], axis=1)
    testset = pd.concat([xTest,yTest], axis=1)


    trainingset.to_csv('train-15.csv',index=False)
    testset.to_csv('test-15.csv',index=False)
    print(trainingset)
    print(testset)


split_train_test()