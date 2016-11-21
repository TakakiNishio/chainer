import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from matplotlib import pyplot as plt


#generate dataset
def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
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
def loss_visualizer(los_data):
    plt.figure(0)
    plt.plot(los_data)
    plt.xlim(0,1000)
    plt.ylim(1e-04,10)
    plt.yscale('log')
    plt.title("LOSS reduction")
    plt.xlabel("epoch")
    plt.ylabel("LOSS")


#visualize regression result
def result_visualizer(x,y):
    plt.figure(1)
    plt.plot(x,np.exp(x),"b",label = "acrual data")
    plt.plot(x, y, "g")
    plt.xlim(0,1)
    plt.ylim(1,3)
    plt.title("regression result")
    plt.xlabel("x")
    plt.ylabel("y")


#main
if __name__ == '__main__':

    model = MyChain()
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)

    losses = []

    epoch = 2000

    for i in range(epoch):

        #generate dataset
        x,y = get_batch(100)

        #set variables
        x_ = Variable(x.astype(np.float32).reshape(100,1))
        t_ = Variable(y.astype(np.float32).reshape(100,1))

        #learning processes
        model.zerograds()   #reset grads
        loss = model(x_,t_) #calculate LOSS
        loss.backward()     #back prop
        optimizer.update()  #renew each weights

        print "epoch: " + str(i) + " LOSS: " + str(loss.data)
        losses.append(loss.data)

    loss_visualizer(losses)

    x_test = np.linspace(0,1,100) #range: 0-1, amount: 100
    y_test = model.get(x_test)

    result_visualizer(x_test,y_test)

    plt.show()
