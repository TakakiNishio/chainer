import numpy as np
import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers

if __name__ == '__main__':

    model = F.Linear(3,3) #define model

    data = np.array([[1,2,3]] , dtype = np.float32) #define input vector
    x = Variable(np.array(data))

    y = model(x) #input x to model (output:y)

    #print(y.data)
    print y.data
