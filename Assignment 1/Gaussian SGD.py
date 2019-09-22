#!/bin/python3.6
#Minyonug Na HW 2
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange
LAMBDA = .001
NUM_SAMP = 250
BATCH_SIZE = 50
NUM_BATCHES = 100
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
        self.y1 = np.float32((-2-(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.sin(self.t))

        self.x2 = np.float32((2+(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.cos(self.t) + 3)
        self.y2 = np.float32((2+(5.5+np.random.normal(0,sigma,num_samp))*self.t) * np.sin(self.t) -3 ) 

        self.setone = np.concatenate((self.x1,self.y1),axis=0)
        self.settwo = np.concatenate((self.x2,self.y2),axis=0)

        self.coord= np.transpose(np.vstack((self.setone,self.settwo)))
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
        activ1 = tf.nn.relu(layer1)  
        layer2 = tf.matmul(activ1, self.w2) + self.b2
        activ2 = tf.nn.relu(layer2)
        layer3 = tf.matmul(activ2,self.w3) + self.b3
        #more layers to come
        return tf.squeeze(layer3)

class Draw():
    def __init__(self):
        fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=[8,8])

        #Display Boundary
        xvalues = np.array([0, 1, 2, 3, 4])
        yvalues = np.array([0, 1, 2, 3, 4])
        xx, yy = np.meshgrid(xvalues, yvalues)
        plt.plot(xx, yy, marker='.', color='k', linestyle='none')
        
        #Display example Data
        plt.scatter(data.x1,data.y1,label='Data 1')
        plt.scatter(data.x2,data.y2,label='Data 2')

        axarr.set_title('Binary Classification')
        axarr.set(xlabel='x', ylabel='y')

        plt.legend()
    
        plt.show()

def loss(mlp, x, y):
  y_ = mlp(x)
  return tf.nn.sigmoid_cross_entropy_with_logits(labels=activations, logits=y_) + (LAMBDA*tf.norm(mlp.w1)**2)

if __name__ == "__main__":
    data = Data()
    mlp=MLP()

    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        
        with tf.GradientTape() as tape:
            coords, activations = data.get_batch()
            loss = (mlp,coords,activations)
            
        grads = tape.gradient(loss, mlp.variables)
        optimizer.apply_gradients(zip(grads, mlp.variables))
        bar.refresh()
    
    
    """ coords_test,activation_test= data.get_batch()
    y_hat = mlp(coords_test)
    loss2= tf.sigmoid(y_hat) 
    print(np.round(loss2)) """

    draw = Draw()

    
