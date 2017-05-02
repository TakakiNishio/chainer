import numpy as np

#chainer library
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import reporter

def pick_up_t(t_data):
    t = []
    for i in range(len(t_data)):
        t.append(t_data[i])
    t = np.array(t,dtype=np.int32)
    return t

#Network definition
class MLP(chainer.Chain):

    def __init__(self, train= True):
        super(MLP, self).__init__(
            l1 = L.Linear(784, 100),
            l2 = L.Linear(100, 100),
            l3 = L.Linear(100, 10)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3


class MLP_classification(chainer.Chain):

    def __init__(self, train= True):
        super(MLP_classification, self).__init__(
            l1 = L.Linear(784, 100),
            l2 = L.Linear(100, 100),
            l3 = L.Linear(100, 10)
        )
        self.train = train

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3

    def __call__(self, x, t):
        self.clear()
        h=self.forward(x)
        #print h.data
        #print t.data
        to = pick_up_t(t.data)
        self.loss = F.softmax_cross_entropy(h, to)
        reporter.report({'loss': self.loss}, self)
        #self.loss = F.mean_squared_error(h, t)
        #print "Loss : " +str(self.loss.data)
        self.accuracy = accuracy.accuracy(h, to)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
