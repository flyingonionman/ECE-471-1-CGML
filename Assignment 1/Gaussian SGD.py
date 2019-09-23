@@ -0,0 +1,114 @@
#!/bin/python3.6
#Minyonug Na HW 1
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange

NUM_FEATURES = 10
NUM_SAMP = 50
BATCH_SIZE = 40
NUM_BATCHES = 300
LEARNING_RATE = 0.1

class Data(object):
    def __init__(self, num_features=NUM_FEATURES, num_samp=NUM_SAMP):
        """
        """
        num_samp = NUM_SAMP
        sigma = 0.1
        np.random.seed(31415)

        #Initial distribution of data
        self.index = np.arange(num_samp)
        self.x = np.random.uniform(size=(num_samp))
        self.y = np.sin(2*np.pi*self.x) + np.random.normal(0,sigma,num_samp)
    
    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)
        return self.x[choices], self.y[choices].flatten()


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        """
        Setup Variables
        w : Weight 
        mu :
        sig : 
        b : Bias
        """
        self.w = tf.Variable(tf.random.normal(shape=[num_features+1, 1]),name='w')
        self.mu = tf.Variable(tf.random.normal(shape=[num_features+1,1]),name='mu')
        self.sig = tf.Variable(tf.random.normal(shape=[num_features+1,1]),name='sig')
        self.b = tf.Variable(tf.zeros(shape=[1, 1]),name='b')

    def __call__(self, x):
        return tf.squeeze(tf.matmul(tf.transpose(self.w),tf.exp(-tf.pow(x-self.mu, 2)/tf.pow(self.sig,2))) + self.b)

class Draw():
    def __init__(self):
        sketch,plotter  = plt.subplots(nrows=1, ncols=2, figsize=[16,8])

        x = np.linspace(0,1)
        y = np.sin(2*np.pi*x)
        plotter[0].plot(x,y,label='Sin wave')
        plotter[0].scatter(data.x,data.y,label='Data')

        plotter[0].set_title("Curve fitting example using SGD")  
        plotter[1].set_title("Gaussians used to curve fit") 

        plotter[0].set(xlabel='x', ylabel='y')
        plotter[1].set(xlabel='x', ylabel='y') 

        x = np.linspace(0,1)
        y= tf.squeeze(tf.matmul(tf.transpose(model.w),tf.exp(-tf.pow(x-model.mu, 2)/tf.pow(model.sig,2))) + model.b)
        plotter[0].plot(x,y, label='Prediction')
        
        """
        Draw individual Gaussians that combine to the final prediction
        """
        for i in range(NUM_FEATURES + 1):
            x = np.linspace(0,1)
            y= tf.exp(-tf.pow(x-model.mu[i], 2)/tf.pow(model.sig[i],2))
            plotter[1].plot(x,y,label = "Gaussian" + str(i)) 
        
        plotter[0].legend()
        plotter[1].legend()

        plt.show()
    
if __name__ == "__main__":
    """
    This part is nearly identical to the linear code; 
    1. Get random batch from the graph
    2. Find y_hat by modeling
    3. Find loss from regression
    4. Alter the variables by looking ta tthe gradient
    """
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    draw = Draw()


    