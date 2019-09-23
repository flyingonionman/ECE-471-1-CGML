#!/bin/python3.6
#Minyonug Na HW 3

#Import basic stuff
import struct as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

#Importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

from keras.optimizers import SGD

#Hyperparams
BATCH_SIZE =128
NUM_CLASSES = 10
EPOCHS = 2
filenames = ['train-labels.idx1-ubyte','t10k-labels.idx1-ubyte','train-images.idx3-ubyte','t10k-images.idx3-ubyte']
class Parse(object):
    def __init__(self) :
        """
        Opens and parses the file 
        I know there there are better ways to open and parse files, but I couldn't manage to do it.
        Very smelly code ahead.
        """
        #Opens up labels
        for i in range(len(filenames)):
            filehandle = open(filenames[i], 'rb')
            filehandle.seek(0)
            if filenames[i][-10:] == "idx1-ubyte" :
                info1 = filehandle.read(8)
                magic,size= st.unpack(">II",info1)
                if filenames[i][:5] == "train" :
                    self.train_label = (st.unpack('>'+'B'*size,filehandle.read(size)))
                else :
                    self.test_label = (st.unpack('>'+'B'*size,filehandle.read(size)))   
            else:
                info2 = filehandle.read(16)
                self.magic, self.size,self.nrow,self.ncol = st.unpack(">IIII", info2)    
                nBytesTotal = self.size*self.nrow*self.ncol*1 #since each pixel data is 1 byte
                if filenames[i][:5] == "train" :
                    self.train_img = np.asarray(st.unpack('>'+'B'*nBytesTotal,filehandle.read(nBytesTotal))).reshape((self.size,self.nrow,self.ncol))
                else :
                    self.test_img = np.asarray(st.unpack('>'+'B'*nBytesTotal,filehandle.read(nBytesTotal))).reshape((self.size,self.nrow,self.ncol))

class cnn(object):
    def __init__(self):
        placeholder =1
if __name__ == "__main__":
    list_val =[]
    list_test=[]
    parse = Parse()
    # img of batch ~ testing
    
    x_train = parse.train_img[0:50000]
    x_val = parse.train_img[50000:60000]
    x_test = parse.test_img

    y_train = parse.train_label[0:50000]
    y_val = parse.train_label[50000:60000]
    y_test = parse.test_label

    #Actually training
    #Apply dropout
    #Apply early stopping
    #Change the loss function
    #Change the L2 penalty
    
    input_shape = (1, parse.nrow, parse.ncol)

    x_train = x_train.reshape(x_train.shape[0], 1, parse.nrow, parse.ncol)
    x_val = x_val.reshape(x_val.shape[0], 1, parse.nrow, parse.ncol)
    x_test = x_test.reshape(x_test.shape[0], 1, parse.nrow, parse.ncol)

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape,
                    data_format='channels_first',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    #Tried to fiddle with the optimizer here. Adadelta was taking way too long per epoch (80 seconds !)
    #This was initially very very innacurate but it seemed that lowering the learning rate ( initially 0.01) drastically improved the results.
    #Takes around 30 second per epoch now.
    sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

    #Trying on validation set
    history = model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_val, y_val))

    score = model.evaluate(x_val, y_val, verbose=0)

    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])

    actualscore = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', actualscore[0])
    print('Test accuracy:', actualscore[1])

    """ fig, axarr = plt.subplots(nrows=1, ncols=4, figsize=[8,4])
    fig.tight_layout()

    for i in range(4):
        axarr[i].imshow(parse.train_img[i],cmap=cm.Greys)
        axarr[i].set_title('Truth=%s' % str(parse.train_label[i]))

    plt.figure()
    plt.plot(history.history['accuracy'],'b-')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show() """

    


    
