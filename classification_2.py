import numpy

#chainer libraty
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers

from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
#from chainer.functions.loss.mean_squared_error import mean_squared_error
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy



class SmallClassificationModel(chainer.FunctionSet):
    def __init__(self):
        super(SmallClassificationModel, self).__init__(
            fc1 = chainer.functions.Linear(2, 2)
        )

    def _forward(self, x):
        h = self.fc1(x)
        return h

    def train(self, x_data, y_data):
        x = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32), volatile=False)
        y = chainer.Variable(y_data.astype(numpy.int32), volatile=False)
        h = self._forward(x)

        optimizer.zero_grads()
        error = chainer.functions.softmax_cross_entropy(h, y)
        error.backward()
        optimizer.update()

        print("x: {}".format(x.data))
        print("h: {}".format(h.data))
        print("h_class: {}".format(h.data.argmax()))

model = SmallClassificationModel()
optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model.collect_parameters())


data_and = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([0])],
    [numpy.array([1,0]), numpy.array([0])],
    [numpy.array([1,1]), numpy.array([1])],
]*1000

for invec, outvec in data_and:
        model.train(invec, outvec)
