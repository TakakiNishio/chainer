#chainer library
import chainer
import chainer.functions as F
import chainer.links as L

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
