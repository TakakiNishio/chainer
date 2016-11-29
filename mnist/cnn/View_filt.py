#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重み取り出してフィルタ可視化
"""
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
import matplotlib.pyplot as plt
import numpy as np

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


def main():
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

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    model = L.Classifier(CNN())
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    i_cw1 = model.predictor.conv1.W.data

    # print(cw1)
    # draw_fil(i_cw1,5)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)
    #print(len(test[0][0][0]))

    i_cy1 = model.predictor.conv1(test[0][0][0].reshape(1, 1, 28, 28)).data
    # draw_cout(i_cy1, len(i_cy1[0][0]))

    samp = test[19]
    sample_la = samp[1]
    print(sample_la)

    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)


    sample = samp[0][0]

    cw1 = model.predictor.conv1.W.data
    cw2 = model.predictor.conv2.W.data
    #y_1 = model.predictor(test[0][0][0].reshape(1,1,28,28))
    cy1 = model.predictor.conv1(sample.reshape(1,1,28,28)).data
    c2y1 = model.predictor.conv2(F.max_pooling_2d(F.relu(cy1),2,stride = 2)).data
    print(cw1.shape)
    print(c2y1.shape)

    ##スケーリング（いらんかった）
    # vmin = cw1.min()
    # vmax = cw1.max()
    # cw1 = (cw1 -vmin).astype(float) / (vmax - vmin).astype(float)
    # #print(cw1)
    # cw1 = cw1*255
    # cw1 = cw1.astype(int)
    ##

    draw_fil(i_cw1, 5)
    draw_raw(sample)
    draw_fil(cw1,5)
    draw_fil(cw2, len(cw2[0][0]))
    draw_cout(cy1, len(cy1[0][0]))
    draw_cout(c2y1,len(c2y1[0][0]))

    plt.show()

def draw_raw(data,size=28):#元画像描写
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(8, 8))
    plt.gray()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.tick_params(labelbottom="off")
    ax.tick_params(labelleft="off")
    xx = np.array(range(size + 1))
    yy = np.array(range(size, -1, -1))
    X, Y = np.meshgrid(xx, yy)
    ax.pcolor(X,Y,data)

def draw_fil(data,size):#フィルタ描写
    plt.style.use('fivethirtyeight')
    tate = len(data)
    yoko = len(data)/21 + 1
    fig = plt.figure(figsize=(tate+2, yoko))
    for t, x in enumerate(data):
        plt.gray()
        ax = fig.add_subplot(yoko , 20, t+1 )
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.tick_params(labelbottom="off")
        ax.tick_params(labelleft="off")
        xx = np.array(range(size + 1))
        yy = np.array(range(size, -1, -1))
        X, Y = np.meshgrid(xx, yy)
        ax.pcolor(X,Y,x[0])

def draw_cout(data,size):#フィルタ適用後出力描写
    plt.style.use('fivethirtyeight')
    tate = len(data[0])
    yoko = len(data[0])/21 + 1
    fig = plt.figure(figsize=(tate+2, yoko))
    for t, x in enumerate(data[0]):
        plt.gray()
        ax = fig.add_subplot(yoko , 20, t+1 )
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.tick_params(labelbottom="off")
        ax.tick_params(labelleft="off")
        xx = np.array(range(size+1))
        yy = np.array(range(size,-1,-1))
        X,Y = np.meshgrid(xx,yy)
        ax.pcolor(X,Y,x)

# def draw_digit_w1(data, n):
#     size = 5
#     #plt.subplot(10, 20, n)_
#     plt.xlim(0,size)
#     plt.ylim(0,size)
#     for c in data:
#         plt
#
#     plt.gray()
#     plt.tick_params(labelbottom="off")
#     plt.tick_params(labelleft="off")

if __name__ == '__main__':
    main()
