import numpy as np
import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers

if __name__ == '__main__':

    #define model
    model = F.Linear(3,3)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    #epoch
    N = 50

    #define input vector
    x = Variable(np.array([[1,2,3]] , dtype = np.float32))

    #correct output
    y = Variable(np.array([[2,4,6]] , dtype = np.float32))

    #learning loop
    for i in range(0,N):

        optimizer.zero_grads()

        theta = model(x) #calcurate output
        print theta.data

        loss = F.mean_squared_error(theta,y) #calcurate error

        loss.backward()    #learning
        optimizer.update()
