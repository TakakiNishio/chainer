#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *

import pylab


# Network definition
class CNN(chainer.Chain):

    def __init__(self, train= True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            fc=L.Linear(800, 500),
            out=L.Linear(500, 10),
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)),2,stride = 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)),2,stride = 2)
        h3 = self.fc(F.dropout(h2))
        return self.out(h3)


def raw_visualiser1(data,label):

    size = 28
    data = np.array(data,dtype = np.float32)
    data = data.reshape(28,28)

    fig1 = plt.figure(1)
    plt.gray()
    plt.xlim(0, 28)
    plt.ylim(0, 28)
    plt.tick_params(labelbottom = 'off')
    plt.tick_params(labelleft = 'off')
    plt.title('%i' % label)
    xx = np.array(range(size + 1))
    yy = np.array(range(size, -1, -1))
    X, Y = np.meshgrid(xx, yy)
    plt.pcolor(X,Y,data)


def dataset_separator(dataset):

    data = []
    label = []

    for i in range(len(dataset)):
        data.append(dataset[i][0])
        label.append(dataset[i][1])

    data = np.array(data,dtype = np.float32)
    label = np.array(label,dtype = np.float32)

    return data,label


def raw_visualiser2(data,label):

    data_amount = len(data)
    p = np.random.random_integers(0, len(data),26)
    fig2 = plt.figure(2)
    plt.gray()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    for i in range(0,25):

        data_i = data[p[i]]
        data_i = data_i.reshape(28,28)

        label_i = label[p[i]]

        ax = fig2.add_subplot(5, 5, i+1)
        ax.set_xlim(0, 28)
        ax.set_ylim(0, 28)
        ax.tick_params(labelbottom="off")
        ax.tick_params(labelleft="off")
        ax.set_title('%i' % label_i)
        xx = np.array(range(28 + 1))
        yy = np.array(range(28, -1, -1))
        X, Y = np.meshgrid(xx, yy)
        ax.pcolor(X,Y,data_i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--initmodel', '-m', default='cnnm.model',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    # print 'GPU: {}'.format(args.gpu))
    # print '# Minibatch-size: {}'.format(args.batchsize))
    # print

    model = L.Classifier(CNN())

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    train_data,train_label = dataset_separator(train)
    test_data,test_label = dataset_separator(test)

    #raw_visualiser1(train_data[0],train_label[0])
    raw_visualiser2(test_data,test_label)

    estimated_label = model.predictor(test_data[0].reshape(1,1,28,28)).data
    print estimated_label
    print np.argmax(estimated_label)
    print test_label[0]

    plt.show()
