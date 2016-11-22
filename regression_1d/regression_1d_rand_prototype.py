import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from matplotlib import pyplot as plt

import random


#define function
def real_function(x):
    #return np.exp(x)
    return -x**2
    #return np.sin(x)


#generate dataset
def dataset_generator(n):
    max_x = 1
    min_x = -1
    #max_x = np.pi
    #min_x = -np.pi
    error_range = 0.05
    x = []
    y = []
    for i in range(n):
        x.append(random.uniform(min_x,max_x))
        #y.append(real_function(x[i]))
        y.append(real_function(x[i]) + random.uniform(-error_range,error_range))
    x = np.array(x)
    y = np.array(y)
    return x,y


#define NN class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(1,16),  #input:1  output:16
            l2 = L.Linear(16,32), #input:16 output:32
            l3 = L.Linear(32,1),  #input:32 output:1
        )

    def __call__(self, x,t): #calculate error
        return F.mean_squared_error(self.predict(x),t)

    def predict(self, x): #calculate network output
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        return h3

    def get(self,x): #confirm tearning result
        return self.predict(Variable(np.array([x]).astype(np.float32).reshape(len(x),1))).data


#visualize loss reduction
def loss_visualizer(train_los_data):
    plt.figure(1)
    plt.plot(train_los_data)
    plt.xlim(0,1000)
    plt.ylim(1e-04,10)
    plt.yscale('log')
    plt.title("train LOSS reduction")
    plt.xlabel("epoch")
    plt.ylabel("train LOSS")


#visualize regression result
def result_visualizer(actual_x,actual_y,test_x,test_y):
    plt.figure(2)
    plt.plot(actual_x,actual_y,"bo",label = "actual data")
    plt.plot(test_x, test_y,"go",label = 'estimated data')
    plt.title("regression result")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")


#main
if __name__ == '__main__':

    data_n = 1000
    train_n = 500
    epoch_n = 2000
    batchsize = 10

    #generate dataset
    dataset = dataset_generator(data_n)
    x_data = dataset[0]
    y_data = dataset[1]

    model = MyChain()
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)

    train_losses = []

    for epoch in range(epoch_n):

        #separate dataset
        i = random.randint(0,train_n-batchsize-1)
        train_x = x_data[i:i+batchsize]
        train_y = y_data[i:i+batchsize]

        #set variables
        train_x_ = Variable(train_x.astype(np.float32).reshape(len(train_x),1))
        train_y_ = Variable(train_y.astype(np.float32).reshape(len(train_y),1))

        #learning processes
        model.zerograds()   #reset grads
        train_loss = model(train_x_,train_y_) #calculate LOSS
        train_loss.backward()     #back propagation
        optimizer.update()  #renew each weights

        print "epoch: " + str(epoch) + "train LOSS: " + str(train_loss.data)
        train_losses.append(train_loss.data)

    loss_visualizer(train_losses)

    test_x = x_data[train_n-1:data_n-1]
    test_y = model.get(test_x)

    result_visualizer(x_data,y_data,test_x,test_y)

    plt.show()
