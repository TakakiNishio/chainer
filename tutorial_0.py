#Matplotlib and NumPy
import numpy as np
import matplotlib.pyplot as plt

#Chainer Specific
from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

if __name__ == '__main__':

    # Create 2 chainer variables then sum their squares
    # and assign it to a third variable.
    a = Variable(np.array([3], dtype=np.float32))
    b = Variable(np.array([4], dtype=np.float32))
    c = a**2 + b**2

    # Inspect the value of your variables.
    print("a.data: {0}, b.data: {1}, c.data: {2}".format(a.data, b.data, c.data))

    # Now call backward() on the sum of squares.
    c.backward()

    # And inspect the gradients.
    print("dc/da = {0}, dc/db = {1}, dc/dc = {2}".format(a.grad, b.grad, c.grad))
