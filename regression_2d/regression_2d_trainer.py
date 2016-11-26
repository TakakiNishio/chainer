import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

import os
import json
import pickle

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers,serializers,utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.mean_squared_error import mean_squared_error

import chainer.datasets.tuple_dataset as td
from chainer import training
from chainer.training import extensions


#define function
def real_function(x1,x2):
    z = -3*np.exp(-(((x1-2)**2)/3)-(((x2-2)**2)/3)) - 4*np.exp(-(((x1+2)**2)/4)-(((x2 +2)**2)/4))
    #z = np.exp(-0.25 * np.sqrt(x1**2 + x2**2)) * np.cos(2 * np.sqrt(x1**2 + x2**2))
    return z


#generate dataset
def dataset_generator(n):

    #define domains
    max_x1 = 5
    min_x1 = -5
    max_x2 = 5
    min_x2 = -5

    error_range = 0.5

    x = []
    y = []

    for i in range(n):
        x1 = random.uniform(min_x1,max_x1)
        x2 = random.uniform(min_x2,max_x2)
        x.append([x1,x2])
        y.append(real_function(x1,x2))
        #y.append(real_function(x1,x2)+random.uniform(-error_range,error_range))

    x = np.array(x, dtype = np.float32)
    y = np.array(y, dtype = np.float32)

    x = np.reshape(x,(len(x),2))
    y = np.reshape(y,(len(y),1))

    return x,y


#define NN class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(2,16),
            l2 = L.Linear(16,32),
            l3 = L.Linear(32,48),
            l4 = L.Linear(48,1)
        )

    def __call__(self, x): #calculate network output
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 = F.leaky_relu(self.l4(h3))
        return h4


#visualize loss reduction
def loss_visualizer():

    epoch = []
    train_loss = []
    test_loss = []

    f = open('./result/log', 'r')
    data = json.load(f)
    f.close()

    # print data
    # print type(data)
    # print len(data)

    value = []

    for i in range(0,len(data)):
        value = data[i]
        epoch.append(value["epoch"])
        train_loss.append(value["main/loss"])
        test_loss.append(value["validation/main/loss"])

    plt.figure(1)
    plt.plot(epoch,train_loss,"b",label = "train LOSS")
    plt.plot(epoch,test_loss,"g",label = "test LOSS")
    #pli.xlim(0,1000)
    #plt.ylim(1e-04,1)
    plt.yscale('log')
    plt.title("LOSS reduction")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("LOSS")


#visualize regression result in draw 3-D graph
def test_result_visualizer(actual_x,actual_y,test_x,estimated_y):

    actual_x1=[]
    actual_x2=[]
    test_x1 = []
    test_x2 = []

    for i in range (len(actual_x)):
        actual_x1.append(actual_x[i][0])
        actual_x2.append(actual_x[i][1])

    for j in range (len(test_x)):
        test_x1.append(test_x[j][0])
        test_x2.append(test_x[j][1])

    fig2 = plt.figure(2)
    ax = Axes3D(fig2)
    p1 = ax.scatter3D(actual_x1,actual_x2,actual_y,color=(1.0,0,0),marker='o',s=10,label='actual data')
    p2 = ax.scatter3D(test_x1,test_x2,estimated_y,color=(0,0,1.0),marker='o',s=10,label='estimated data')

    plt.title("regression for test dataset")
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")


#draw 3D graph
def function_visualizer():

    x1_mesh = np.arange(-5,5,0.25)
    x2_mesh = np.arange(-5,5,0.25)

    X1,X2 = np.meshgrid(x1_mesh,x2_mesh)
    X_mesh =[]

    actual_Z = real_function(X1,X2)

    for i in range(0,len(x1_mesh)):
        for j in range(0,len(x2_mesh)):
            X_mesh.append([x1_mesh[i],x2_mesh[j]])

    X_mesh = np.array(X_mesh,dtype = np.float32)
    estimated_Z = np.array((model.predictor(X_mesh).data).reshape((len(x1_mesh),len(x2_mesh))))
    fig3 = plt.figure(figsize = plt.figaspect(0.5))
    ax1 = fig3.add_subplot(1, 2, 1, projection='3d')
    #ax1 = Axes3D(fig3)
    w1 = ax1.plot_wireframe(X1,X2,actual_Z,color=(1.0,0,0),label='actual function')
    ax1.legend()
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")

    #fig4 = plt.figure(4)
    #ax2 = Axes3D(fig4)
    ax2 = fig3.add_subplot(1, 2, 2, projection='3d')
    w2 = ax2.plot_wireframe(X1,X2,estimated_Z,color=(0,0,1.0),label='estimated function')
    ax2.legend()
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("y")


#main
if __name__ == '__main__':

    train_n = 10000
    test_n = 1000

    epoch_n = 1000
    batchsize = 10

    train_x, train_y = dataset_generator(train_n)
    test_x, test_y = dataset_generator(test_n)

    model = L.Classifier(MyChain(), lossfun = mean_squared_error)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)

    start_time = time.time()

    train = td.TupleDataset(train_x,train_y)
    test = td.TupleDataset(test_x,test_y)

    train_iter = chainer.iterators.SerialIterator(train,batchsize)
    test_iter = chainer.iterators.SerialIterator(test,batchsize,repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (epoch_n, 'epoch'), out="result")
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    #trainer.extend(extensions.snapshot())
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    execution_time = time.time() - start_time
    print "execution time : " + str(execution_time)

    #visualize results
    loss_visualizer()
    estimated_y = model.predictor(test_x).data
    test_result_visualizer(test_x,test_y,test_x,estimated_y)
    function_visualizer()
    plt.show()
