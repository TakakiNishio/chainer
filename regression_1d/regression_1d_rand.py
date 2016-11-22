import numpy as np
from matplotlib import pyplot as plt
import random
import time

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


#define function
def real_function(x):
    #return np.exp(x)
    #return -x**2
    return np.sin(2*x)
    #return 2 * x + 1


#generate dataset
def dataset_generator(n):
    #max_x = 1
    #min_x = -1
    max_x = np.pi
    min_x = -np.pi
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
            l1 = L.Linear(1,16),
            l2 = L.Linear(16,32),
            l3 = L.Linear(32,1),
        )

    def __call__(self, x,y): #calculate error
        x_ = Variable(x.astype(np.float32).reshape(len(x),1))
        y_ = Variable(y.astype(np.float32).reshape(len(y),1))
        return F.mean_squared_error(self.predict(x_),y_)

    def predict(self, x): #calculate network output
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        return h3

    def get(self,x): #confirm tearning result
        return self.predict(Variable(np.array([x]).astype(np.float32).reshape(len(x),1))).data


#visualize loss reduction
def loss_visualizer(train_loss_data,test_loss_data):
    plt.figure(1)
    plt.plot(train_loss_data,"b",label = "train LOSS")
    plt.plot(test_loss_data,"g",label = "test LOSS")
    #pli.xlim(0,1000)
    #plt.ylim(1e-04,1)
    plt.yscale('log')
    plt.title("LOSS reduction")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("LOSS")


#visualize regression result
def result_visualizer(actual_x,actual_y,test_x,test_y):
    plt.figure(2)
    plt.plot(actual_x,actual_y,"ro",label = "actual data")
    plt.plot(test_x, test_y,"bo",label = "estimated data")
    #pli.xlim(-1,1)
    #plt.ylim(-1,1)
    plt.title("regression result")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")


#main
if __name__ == '__main__':

    data_n = 500
    train_n = 400

    epoch_n = 300
    batchsize = 10

    dataset = dataset_generator(data_n)
    train_x = dataset[0][0:train_n]
    train_y = dataset[1][0:train_n]
    test_x = dataset[0][train_n+1:data_n-1]
    test_y = dataset[1][train_n+1:data_n-1]

    model = MyChain()
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)

    train_losses = []
    test_losses = []

    start_time = time.time()

    #learning
    for epoch in range(1,epoch_n+1):

        perm = np.random.permutation(train_n)
        sum_loss = 0

        for i in range(0,train_n,batchsize):
            batch_x = train_x[perm[i:i+batchsize]]
            batch_y = train_y[perm[i:i+batchsize]]

            model.zerograds()   #reset grads
            loss = model(batch_x,batch_y)
            sum_loss += loss.data * batchsize
            loss.backward()     #back propagation
            optimizer.update()  #renew each weights

        #calculate train LOSS
        average_loss = sum_loss / data_n
        train_losses.append(average_loss)

        #calculate test LOSS
        test_loss = model(test_x,test_y)
        test_losses.append(test_loss.data)

        if epoch % 10 == 0:
            print "epoch: " + str(epoch) + "  train LOSS: " + str(average_loss) + "  test LOSS: " + str(test_loss.data)

    execution_time = time.time() - start_time
    print "execution time : " + str(execution_time)

    #visualize results
    test_y = model.get(test_x)
    loss_visualizer(train_losses,test_losses)
    result_visualizer(dataset[0],dataset[1],test_x,test_y)
    plt.show()
