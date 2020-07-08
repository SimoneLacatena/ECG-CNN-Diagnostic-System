#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:22:47 2020

@author: martina
"""

import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D,Dense,Activation,BatchNormalization,MaxPooling1D,AveragePooling1D,Dropout,GlobalAveragePooling1D,Flatten
import pandas as pd

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as conf_matrix
from sklearn.metrics import classification_report
import keras.backend as K

def load_ecg(path= "/content/drive/My Drive/training/end_dataset.csv"):
    return pd.read_csv(path)


def showConfusionMatrix(y_test,y_pred):
    
    print("Confusion Matrix")
    matrix = conf_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show() 

    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1),target_names = ['N','A']))

    
    
    return matrix


def mapvalues(y:np.array,mapping:dict):
    y_bin = []
    for y_i in y:
        y_bin.append(mapping[y_i])
    return y_bin






def createCNN(num_unit = 7,initial_kernel_size = 80,strides = 4 ,kernel_initializer = 'normal',maxpooling_poolsize = 4 ,activation= 'relu',avg_poolsize = 2,kernel_size = 3, n_classes=2):
    
    '''layer 1'''
    cnn = Sequential()
    cnn.add(Conv1D(2**num_unit, kernel_size=initial_kernel_size,strides=strides, input_shape=(10800,1)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=maxpooling_poolsize))
    cnn.add(Activation(activation))
    
    
    '''layer 2'''
    cnn.add(Conv1D(2**num_unit,kernel_size=kernel_size))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=maxpooling_poolsize))
    cnn.add(Activation(activation))

    '''layer 3'''
    cnn.add(Conv1D(2**(num_unit+1),kernel_size=kernel_size))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=maxpooling_poolsize))
    cnn.add(Activation(activation))

    '''layer 4'''
    cnn.add(Conv1D(2**(num_unit+2),kernel_size=kernel_size))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=maxpooling_poolsize))
    cnn.add(Activation(activation))

    '''layer 5'''
    cnn.add(AveragePooling1D(pool_size = avg_poolsize))
    cnn.add(Flatten())
    cnn.add(Dense(n_classes, kernel_initializer=kernel_initializer, activation='softmax'))

    return cnn



    
 
def process_ECGDATA(data:pd.DataFrame,test_size=0.3):
   
    y = data['label']
    x=data.drop(["label"],axis=1)
   
 
    #converto on array
    x = np.asarray(x,dtype=np.float32)
    y = np.asarray(y)
    
    y = mapvalues(y,{'N':0,'A':1})
    
    y = np.asarray(y,dtype=np.float32)
    
    print("ECG segment Normal (N) ",num_labels(y,0))
    print("ECG segment Anormal (A) ",num_labels(y,1))
    
    
    xLearning,xTest,yLearning,yTest = train_test_split(x,y,test_size= test_size)
    
    return xLearning,xTest,yLearning,yTest


def num_labels(data,label):
    num = 0
    for value in data:
        if(value == label):
             num += 1
    return num



def validation(CNN_loader,LearningX,LearningY,save_dir,K = 5,test_size = 0.1,batch_size= 128,epochs=50):
    
    #data
    
    n_classes = 2
    
    
    
    LearningX = LearningX.reshape(LearningX.shape[0], LearningX.shape[1],1)

    
    #to categorical
    LearningY = keras.utils.to_categorical(LearningY, n_classes)
    
    
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=K, shuffle=True)
    
    
    # K-fold Cross Validation model evaluation
    
    
    fold_no = 1
    
    ACCURACY=[]
    LOSSLESS=[]
    RECALL = []
  
    for train, val in kfold.split(LearningX,LearningY):
        model = CNN_loader()

        trainingSet_X = LearningX[train]
        trainingSet_y = LearningY[train]

        validation_X  = LearningX[val]
        validation_Y  = LearningY[val]

        print("New Iteration , K = " + fold_no.__str__())
        print("num N ( Normal ) on tuning:" + str(num_labels(validation_Y.argmax(axis=1),0)))
        print("num A (Anormal)  on tuning:" + str(num_labels(validation_Y.argmax(axis=1),1)))

        print("num N ( Normal ) on training:" + str(num_labels(trainingSet_y.argmax(axis=1),0)))
        print("num A (Anormal)  on training:" + str(num_labels(trainingSet_y.argmax(axis=1),1)))
        
        
        
        
        
        # CREATE CALLBACKS
        checpoint_dir = save_dir + '/model_' + str(fold_no) + '.h5'
        checkpoint = ModelCheckpoint(checpoint_dir, monitor='val_recall', verbose=1, save_best_only=True, mode='max')
    
    
        history = model.fit(trainingSet_X, trainingSet_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = (validation_X, validation_Y),
              callbacks = [checkpoint])
        
        
        model.load_weights(checpoint_dir)
        
        loss,acc,recall = model.evaluate( validation_X, validation_Y,batch_size=batch_size, verbose=1)
        ACCURACY.append(acc)
        LOSSLESS.append(loss)
        RECALL.append(recall)
        
        print("Model Results: ")
        print("lossless =  " + str(loss))
        print("accuracy = " + str(acc))
        print("recall =  " + str(recall))
        
       
    
        # summarize history for loss
        print('Accuracy Graph')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('models')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()

        print('Recall Graph')
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('models')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()
        
        fold_no = fold_no + 1
    
        
    print('mean accuracy %.3f +/- %.3f' % (np.mean(ACCURACY),np.std(ACCURACY)))
    print('mean lossless %.3f +/- %.3f' % (np.mean(LOSSLESS),np.std(LOSSLESS)))
    print('mean recall %.3f +/- %.3f' % (np.mean(RECALL),np.std(RECALL)))
    

def finalPrediction(LearningX,LearningY,TestX,TestY,model, save_dir,batch_size=128,epochs=10):
    n_classes = 2
    LearningX = LearningX.reshape(LearningX.shape[0], LearningX.shape[1],1)
    LearningY = keras.utils.to_categorical(LearningY, n_classes)
    print(TestX.shape)
    
    TestX = TestX.reshape(TestX.shape[0], TestX.shape[1],1)
    TestY = keras.utils.to_categorical(TestY, n_classes)


    #checpoint_dir = save_dir + '/final_model.h5'
    #checkpoint = ModelCheckpoint(checpoint_dir, monitor='val_recall', verbose=1, save_best_only=True, mode='max')

    model.fit(LearningX,LearningY,batch_size=batch_size,epochs=epochs, verbose=1)

    #model.load_weights(checpoint_dir)
    Ypred = model.predict(TestX)
    showConfusionMatrix(TestY,Ypred)
    model.save(save_dir + '/final_model')
    
#k of kflod    
K = 10
    
#Parametres
#-----------------------------------  

BATCH_SIZE = 32  
MAXPOOL_POOL_SIZE = 4
AVGPOOL_POOL_SIZE = 2
KERNEL_SIZE = 4
KERNEL_INITIALIZER = 'normal'
STRIDES = 4
INITIAL_KERNEL_SIZE = 80
NUM_UNIT = 7  # 2^NUM_UNIT  128 unit


LOSS_FUNCTION = 'binary_crossentropy'
ACTIVATION = 'relu'
EPOCHS = 100
OPTIMIZER = 'adam'
#LEARNING_RATE = 0.1

#--------------------------------



def CNN_LOAD():
    model =  createCNN(num_unit = NUM_UNIT,
                       initial_kernel_size = INITIAL_KERNEL_SIZE,
                       strides = STRIDES ,
                       kernel_initializer = KERNEL_INITIALIZER,
                       maxpooling_poolsize = MAXPOOL_POOL_SIZE ,
                       activation= ACTIVATION,
                       avg_poolsize = AVGPOOL_POOL_SIZE,
                       kernel_size = KERNEL_SIZE)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy',tensorflow.keras.metrics.Recall(name='recall')])
    return model



SAVE_DIR = '/content/drive/My Drive/3 AnnoUniversità/engConsoc/sistemi_multimediali'


print('PARAMS: ')
print('epochs = ',EPOCHS)
print('batch_size = ',BATCH_SIZE)  #numero di terazioni per ogni epoca dipnede da questo valore
print('maxpoolsize = ',MAXPOOL_POOL_SIZE)
print('avgpoolsize = ',AVGPOOL_POOL_SIZE)
print('kernel_size = ',KERNEL_SIZE)
print('optimizer = ',OPTIMIZER )
print('activation = ',ACTIVATION)
print('kernel_initializer = ',KERNEL_INITIALIZER)
print('strides = ',STRIDES )
print('num_unit = ',2**NUM_UNIT)
LOSS_FUNCTION = 'binary_crossentropy'
print('initial_kernel_size = ',INITIAL_KERNEL_SIZE)
#print('learnign_rate = ',LEARNING_RATE)







print('loading dataset...')
data=load_ecg("/content/drive/My Drive/3 AnnoUniversità/end_dataset.csv")
print('DONE')

print('process dataset...')
xLearning,xTest,yLearning,yTest=process_ECGDATA(data)
print('DONE')

print("start validation")
#validation(CNN_LOAD,xLearning,yLearning,save_dir=SAVE_DIR,epochs=EPOCHS, K= K,batch_size=BATCH_SIZE)

print("start final prediction")
model = CNN_LOAD()
model.summary()
finalPrediction(xLearning,yLearning,xTest,yTest,model = model,save_dir = SAVE_DIR,epochs=EPOCHS,batch_size=BATCH_SIZE)
 



    
    
          
    
