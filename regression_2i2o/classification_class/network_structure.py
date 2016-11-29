#chainer library
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers,serializers,utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


#define NN class
class MyChain1(Chain):

    def __init__(self):
        super(MyChain1, self).__init__(
            l1 = L.Linear(2,150),
            l2 = L.Linear(150,2)
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        return h


#define NN class
class MyChain2(Chain):

    def __init__(self):
        super(MyChain2, self).__init__(
            l1 = L.Linear(2,10),
            l2 = L.Linear(10,15),
            l3 = L.Linear(15,2)
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


#define NN class
class MyChain3(Chain):

    def __init__(self):
        super(MyChain3, self).__init__(
            l1 = L.Linear(2,10),
            l2 = L.Linear(10,15),
            l3 = L.Linear(15,10),
            l4 = L.Linear(10,2)
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h
