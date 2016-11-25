#chainer library
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers,serializers,utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


#define NN class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(2,16),
            l2 = L.Linear(16,32),
            l3 = L.Linear(32,48),
            l4 = L.Linear(48,1)
        )

    def __call__(self, x): #calculate network output
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 = F.leaky_relu(self.l4(h3))
        return h4
