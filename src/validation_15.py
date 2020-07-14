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

def load_ECGs(path= "/content/drive/My Drive/training/end_dataset-15.csv",propA = 0.4):
    dataset = pd.read_csv(path)
    xLearning,xTest,yLearning,yTest = process_ECGDATA(dataset)
    xTest,yTest = clean(propA,xTest,yTest)
    return xLearning,xTest,yLearning,yTest




def clean(propA,xTest,yTest):
  n=len(xTest)
  numN=num_labels(yTest,0)
  numA=num_labels(yTest,1)
  print('numero di A prima',numA)
  print('numero di N prima',numN)
  aPropN=numN/n
  aPropA=numA/n
  while aPropA>propA :
    for row in range(len(yTest)):
      if yTest[row]==1 :
        xTest=np.delete(xTest, row, 0)
        yTest=np.delete(yTest, row, 0)
        break
    numA=num_labels(yTest,1)
    n-=1
    aPropA=numA/n
  
  print('numero di A Dopo',num_labels(yTest,1))
  print('numero di N Dopo',num_labels(yTest,0))
  print('xTest',xTest)
  print('yTest',yTest)

  return xTest,yTest


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







def createCNN(layer_sizes = (128,255,512),initial_kernel_size = 80,strides = 4 ,padding = 'valid',kernel_initializer = 'normal',maxpooling_poolsize = 4 ,activation= 'relu',avg_poolsize = 2,kernel_size = 3, n_classes=2):
    
    cnn = Sequential()
    cnn.add(Conv1D(layer_sizes[0], kernel_size=initial_kernel_size,strides=strides, padding = padding, input_shape=(5400,1)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling1D(pool_size=maxpooling_poolsize))
    cnn.add(Activation(activation))
    for layer_size in  layer_sizes:
      cnn.add(Conv1D(layer_size,kernel_size=kernel_size))
      cnn.add(BatchNormalization())
      cnn.add(MaxPooling1D(pool_size=maxpooling_poolsize))
      cnn.add(Activation(activation))


    cnn.add(AveragePooling1D(pool_size = avg_poolsize))
    cnn.add(Flatten())
    #cnn.add(Dense(100, activation=activation))
    cnn.add(Dense(n_classes,kernel_initializer = kernel_initializer, activation='softmax'))

    return cnn



    
 



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
    
def saveECGClassifications(xTest,yTrue,yPred,output_dir):
  examples = list(xTest.reshape(xTest.shape[0],xTest.shape[1]))
  yTrue = yTrue.argmax(axis = 1)
  yPred = yPred.argmax(axis = 1)
  example_list = []
  for i in range(len(yTrue)):
    example_list.append(list(examples[i]))
    example_list[i].append(yTrue[i])

  
  pred_0_true_1 = []
  pred_1_true_0 = []
  for  example,y_pred in zip(example_list,yPred):
    y_true = example[-1]
    if y_true != y_pred:
      if y_pred == 1 and y_true == 0:
        pred_1_true_0.append(example)
      else : 
        pred_0_true_1.append(example)
  pd.DataFrame(data = pred_0_true_1,).to_csv(output_dir+ '/predN_trueA.csv',index =False)
  pd.DataFrame(data = pred_1_true_0,).to_csv(output_dir+ '/predA_trueN.csv',index =False)





def finalPrediction(LearningX,LearningY,TestX,TestY,model, save_dir,batch_size=128,epochs=10):
    n_classes = 2
    LearningX = LearningX.reshape(LearningX.shape[0], LearningX.shape[1],1)
    LearningY = keras.utils.to_categorical(LearningY, n_classes)
    print(TestX.shape)
    
    TestX = TestX.reshape(TestX.shape[0], TestX.shape[1],1)
    TestY = keras.utils.to_categorical(TestY, n_classes)


    checpoint_dir = save_dir + '/final_model.h5'
    checkpoint = ModelCheckpoint(checpoint_dir, monitor='val_recall', verbose=1, save_best_only=True, mode='max')

    model.fit(LearningX,LearningY,batch_size=batch_size,validation_split = 0.1,epochs=epochs, verbose=1,callbacks = [checkpoint])

    model.load_weights(checpoint_dir)
    Ypred = model.predict(TestX)
    showConfusionMatrix(TestY,Ypred)
    model.save(save_dir + '/final_model')
    saveECGClassifications(TestX,TestY,Ypred,SAVE_DIR)

#k of kflod    
K = 10
    
#Parametres
#-----------------------------------  

BATCH_SIZE = 16
MAXPOOL_POOL_SIZE = 4
AVGPOOL_POOL_SIZE = 2

KERNEL_SIZE = 4
STRIDES = 6
INITIAL_KERNEL_SIZE = 35
LAYER_SIZES = (64,86,128)

KERNEL_INITIALIZER = 'normal'
LOSS_FUNCTION = 'binary_crossentropy'
ACTIVATION = 'relu'
EPOCHS = 40
OPTIMIZER = 'Adamax'
PADDING = 'valid'


#--------------------------------



def CNN_LOAD():
    model =  createCNN(layer_sizes = LAYER_SIZES,
                       initial_kernel_size = INITIAL_KERNEL_SIZE,
                       strides = STRIDES ,
                       kernel_initializer = KERNEL_INITIALIZER,
                       maxpooling_poolsize = MAXPOOL_POOL_SIZE ,
                       activation= ACTIVATION,
                       avg_poolsize = AVGPOOL_POOL_SIZE,
                       kernel_size = KERNEL_SIZE,
                       padding = PADDING)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy',tensorflow.keras.metrics.Recall(name ='recall')])
    return model



SAVE_DIR = '/content/drive/My Drive/training/last_experiment'


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
print('units = ',LAYER_SIZES)
LOSS_FUNCTION = 'binary_crossentropy'
print('initial_kernel_size = ',INITIAL_KERNEL_SIZE)







print('loading dataset...')
d = xLearning,xTest,yLearning,yTest = load_ECGs()
print(d)
print('DONE')

print("start validation")
validation(CNN_LOAD,xLearning,yLearning,save_dir=SAVE_DIR,K = K,epochs=EPOCHS,batch_size=BATCH_SIZE)

print("start final prediction")
model = CNN_LOAD()
model.summary()
finalPrediction(xLearning,yLearning,xTest,yTest,model = model,save_dir = SAVE_DIR,epochs=EPOCHS,batch_size=BATCH_SIZE)
 




    
          
    
