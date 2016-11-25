#python library
import time
from matplotlib import pyplot as plt

#chainer library
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers,serializers,utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.mean_squared_error import mean_squared_error

import chainer.datasets.tuple_dataset as td
from chainer import training
from chainer.training import extensions

#python scripts
import get_dataset as d
import network_structure as nn
import visualizer as v

#main
if __name__ == '__main__':

    #set parameters
    train_n = 5000
    test_n = 1000
    epoch_n = 50
    batchsize = 10

    #get dataset
    train_x, train_y = d.dataset_generator(train_n)
    test_x, test_y = d.dataset_generator(test_n)

    #load and setup the network
    model = L.Classifier(nn.MyChain(), lossfun = mean_squared_error)
    model.compute_accuracy = False #for regression
    optimizer = optimizers.Adam()  #choose optimizer
    optimizer.setup(model)

    start_time = time.time() #start time measurement

    #modify dataset structure
    train = td.TupleDataset(train_x,train_y)
    test = td.TupleDataset(test_x,test_y)

    #setup iterator
    train_iter = chainer.iterators.SerialIterator(train,batchsize)
    test_iter = chainer.iterators.SerialIterator(test,batchsize,repeat=False, shuffle=False)

    #trainer
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

    serializers.save_npz('my.model', model)
    serializers.save_npz('my.state', optimizer)
    #visualize results
    v.loss_visualizer()
    estimated_y = model.predictor(test_x).data
    v.test_result_visualizer(test_x,test_y,test_x,estimated_y)
    # v.function_visualizer(model)
    plt.show()
