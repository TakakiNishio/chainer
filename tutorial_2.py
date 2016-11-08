#Matplotlib and NumPy
import numpy as np
import matplotlib.pyplot as plt

#Chainer Specific
from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L


# Define a forward pass function taking the data as input.
# and the linear function as output.
def linear_forward(data):
    return linear_function(data)


# Define a training function given the input data, target data,
# and number of epochs to train over.
def linear_train(train_data, train_target,n_epochs=200):
    for _ in range(n_epochs):
        # Get the result of the forward pass.
        output = linear_forward(train_data)

        # Calculate the loss between the training data and target data.
        loss = F.mean_squared_error(train_target,output)

        # Zero all gradients before updating them.
        linear_function.zerograds()

        # Calculate and update all gradients.
        loss.backward()

        # Use the optmizer to move all parameters of the network
        # to values which will reduce the loss.
        optimizer.update()


if __name__ == '__main__':

    # Generate linearly related datasets x and y.
    x = 30*np.random.rand(1000).astype(np.float32)
    y = 7*x+10
    y += 10*np.random.randn(1000).astype(np.float32)

    #plt.scatter(x,y)
    #plt.xlabel('x')
    #plt.ylabel('y')

    # Setup linear link from one variable to another.
    linear_function = L.Linear(1,1)

    # Set x and y as chainer variables, make sure to reshape
    # them to give one value at a time.
    x_var = Variable(x.reshape(1000,-1))
    y_var = Variable(y.reshape(1000,-1))

    # Setup the optimizer.
    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(linear_function)

    # This code is supplied to visualize your results.
    plt.scatter(x,y, alpha =0.5)

    for i in range(150):
        linear_train(x_var, y_var, n_epochs=5)
        y_pred = linear_forward(x_var).data
        plt.plot(x, y_pred, color=plt.cm.cool(i/150.), alpha = 0.4, lw =3)


    slope = linear_function.W.data[0,0]
    intercept = linear_function.b.data[0]
    plt.title("Final Line: {0:.3}x + {1:.3}".format(slope, intercept))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
