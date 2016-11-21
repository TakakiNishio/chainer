import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from matplotlib import pyplot as plt

#generate data
def get_bach(n):
    x = np.random.random(n) # n random numbers(0-1.0)
    y = np.exp(x)
    return x,y


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(1,16),
            l2 = L.Linear(16,32),
            l3 = L.Linear(32,1),
        )

    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x),t)

    def predict(self,x):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        return h3

    def get(self,x):
        return self.predict(Variable(np.array([x]).astype(np.float32).reshape(1,1)))



if __name__ == "__main__":

    #print get_bach(2)
    
