#Matplotlib and NUmpy
import numpy as np
import matplotlib.pyplot as plt

#Chainer Specific
from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L


def linear_forward(data):
    return linear_function(data)


def linear_train(train_data, train_target, n_epochs=200):
    for i in range(n_epochs):
        output = linear_forward(train_data)
        loss = F.mean_squared_error(train_target, output)
        linear_function.zerograds()
        loss.backward()
        optimizer.update()


if __name__ == '__main__':

    x = 30*np.random.rand(1000).astype(np.float32)
    y = 7*x + 10
    y += 10*np.random.rand(1000).astype(np.float32)

    linear_function = L.Linear(1,1)

    x_var = Variable(x.reshape(1000,-1))
    y_var = Variable(y.reshape(1000,-1))

    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(linear_function)

    plt.scatter(x,y,alpha=0.5)

    for i in range(150):
        linear_train(x_var, y_var, n_epochs=5)
        y_pred = linear_forward(x_var).data
        plt.plot(x, y_pred, color=plt.cm.cool(i/150.),alpha=0.4,lw=3)

    slope = linear_function.W.data[0,0]
    intercept = linear_function.b.data[0]
    plt.title("Final Line: {0:.3}x + {1:.3}".format(slope, intercept))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
