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
    return np.exp(x)

#generating dataset
def dataset_generator(n):
    error_range = 0.05
    x = []
    y = []
    for i in range(n):
        x.append(np.random.random())
        y.append(real_function(x[i])
        #y.append(real_function(x[i]) + random.uniform(-error_range,error_range))
    return x,y

if __name__ == '__main__':

    dataset = dataset_generator(500)

    plt.figure(1)
    plt.plot(dataset[0],dataset[1],"bo",label = "dataset")
    plt.xlim(0,1)
    plt.ylim(1,3)
    plt.title("exp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    #print dataset[0]
