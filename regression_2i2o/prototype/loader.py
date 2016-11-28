#python library
import numpy as np
from matplotlib import pyplot as plt

#chainer library
import chainer
#from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
#import chainer.functions as F
import chainer.links as L
#from chainer.training import extensions
from chainer.functions.loss.mean_squared_error import mean_squared_error

#python scripts
import network_structure as nn
import visualizer as v

#main
if __name__ == '__main__':

    model = L.Classifier(nn.MyChain1(), lossfun = mean_squared_error)
    #model = L.Classifier(nn.MyChain2(), lossfun = mean_squared_error)
    #model = L.Classifier(nn.MyChain3(), lossfun = mean_squared_error)
    model.compute_accuracy = False #for regression
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)
    serializers.load_npz('my.model', model)
    #serializers.load_npz('my.state', optimizer)

    v.function_visualizer(model)
    plt.show()
