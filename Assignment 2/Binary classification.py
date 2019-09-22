#!/bin/python3.6
#Minyonug Na HW 2
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange
LAMBDA = .01
NUM_SAMP = 250
BATCH_SIZE = 32
NUM_BATCHES = 100000
LEARNING_RATE = 0.01

class Data(object):
    def __init__(self, num_samp=NUM_SAMP):
        num_samp = NUM_SAMP
        sigma = .1
        np.random.seed(31415)
        #Initial distribution of data
        self.index = np.arange(num_samp*2)

        self.t = np.random.uniform(size=(num_samp))* 3.5 *np.pi 
        self.x1 = np.float32((-2-(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.cos(self.t))
        self.x2 = np.float32((2+(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.cos(self.t) + 3)

        self.y1 = np.float32((-2-(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.sin(self.t))
        self.y2 = np.float32((2+(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.sin(self.t) -3 ) 

        self.setone = np.transpose(np.vstack((self.x1,self.y1)))
        self.settwo = np.transpose(np.vstack((self.x2,self.y2)))

        self.coord= np.vstack((self.setone,self.settwo))
        self.activation = np.concatenate((np.float32(np.zeros(num_samp)),np.float32(np.ones(num_samp))), axis=0)

    def get_batch(self, batch_size=BATCH_SIZE):

        choices = np.random.choice(self.index, size=batch_size)
        return self.coord[choices,:], self.activation[choices]

class MLP(tf.Module):
    def __init__(self):
        # Neural network
        # Use activation function matmul in series

        #Hidden Layer
        
        self.w1 = tf.Variable(tf.random.normal(shape=[2,50],mean=0,stddev=1),name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=[1, 50]),name='b1')
        self.w2 = tf.Variable(tf.random.normal(shape=[50,25]),name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=[1, 25]),name='b2')
        self.w3 = tf.Variable(tf.random.normal(shape=[25,1]),name='w3')
        self.b3 = tf.Variable(tf.zeros(shape=[1, 1]),name='b3')
    def __call__(self, points):
        #Neural Network
        layer1 = tf.matmul(points, self.w1) +self.b1
        activ1 = tf.nn.elu(layer1)  
        layer2 = tf.matmul(activ1, self.w2) + self.b2
        activ2 = tf.nn.elu(layer2)
        layer3 = tf.matmul(activ2,self.w3) + self.b3
        #more layers to come
        return tf.squeeze(layer3)

class Draw():
    def __init__(self,xx,yy,yyy):
        fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=[8,8])

        
        plt.contourf(xx, yy, yyy, [0,.5,1])

        #Display example Data
        plt.scatter(data.x1,data.y1,label='Data 0')
        plt.scatter(data.x2,data.y2,label='Data 1')
        

        axarr.set_title('Binary Classification')
        axarr.set(xlabel='x', ylabel='y')

        plt.legend()
    
        plt.show()

if __name__ == "__main__":
    data = Data()
    mlp=MLP()

    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        
        with tf.GradientTape() as tape:
            coords, activations = data.get_batch()
            y_hat = mlp(coords)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=activations, logits=y_hat) + (LAMBDA*tf.norm(mlp.w1)**2))
        grads = tape.gradient(loss, mlp.variables)
        optimizer.apply_gradients(zip(grads, mlp.variables))
        bar.refresh()
    

    x = y = np.linspace(-60,60,300)
    xf = np.float32(x)
    yf = np.float32(y)

    xx, yy = np.meshgrid(xf,yf)
    allp = np.array(list(zip(xx.flatten(), yy.flatten())))
    y_hat = mlp(allp)
    
    yyy = tf.math.sigmoid(y_hat)
    pleasesendhelp = np.reshape(yyy,(300,300))

    draw = Draw(xx,yy,pleasesendhelp)
    
