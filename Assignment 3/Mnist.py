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

from keras.optimizers import SGD

#Hyperparams
BATCH_SIZE =128
NUM_CLASSES = 10
EPOCHS = 12
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
    parse = Parse()
    # img of batch ~ testing
    fig, axarr = plt.subplots(nrows=1, ncols=4, figsize=[8,4])
    fig.tight_layout()

    for i in range(4):
        axarr[i].imshow(parse.train_img[i],cmap=cm.Greys)
        axarr[i].set_title('Truth=%s' % str(parse.train_label[i]))

    x_train = parse.train_img[0:49999]
    x_test = parse.test_img

    y_train = parse.train_label[0:49999]
    y_test = parse.test_label
    #Actually training
    #Apply dropout
    #Apply early stopping
    #Change the loss function
    #Change the L2 penalty
    
    input_shape = (1, parse.nrow, parse.ncol)

    x_train = x_train.reshape(x_train.shape[0], 1, parse.nrow, parse.ncol)
    x_test = x_test.reshape(x_test.shape[0], 1, parse.nrow, parse.ncol)

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.show()

    


    
