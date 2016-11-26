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

    model = L.Classifier(nn.MyChain(), lossfun = mean_squared_error)
    model.compute_accuracy = False #for regression
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)
    serializers.load_npz('my.model', model)
    #serializers.save_npz('my.state', optimizer)

    # x = []
    # x1 = np.arange(-1,1,0.5)
    # x2 = np.arange(-1,1,0.5)

    # for i in range(0,len(x1)):
    #     for j in range(0,len(x2)):
    #         x.append([x1[i],x2[j]])

    # x = np.array(x,dtype = np.float32)
    # y = model.predictor(x).data
    # print y

    v.function_visualizer(model)
    plt.show()
