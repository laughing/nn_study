#coding: utf-8
import numpy

def sigmoid(u):
    return 1.0 / (1.0 + numpy.exp(-u))

def dsigmoid(y):
    return y * (1.0 - y)

class Layer(object):
    def __init__(self, num_input, num_output, f=None, df=None, nobias=False, rng=None):
        self.f = f
        self.df = df
        self.nobias = nobias
        if rng is None:
            rng = numpy.random.RandomState(1234)
        self.W = rng.normal(
            0, numpy.sqrt(1. / num_input), 
            (num_output, num_input)).astype(numpy.float32)
        
        if not self.nobias:
            self.b = numpy.zeros(num_output, dtype=numpy.float32)

    def forward(self, x):
        u = self.W.dot(x)
        if not self.nobias:
            u += numpy.atleast_2d(self.b).T
        z = self.f(u)
        self.x = x
        self.z = z
        return z

    def backward(self, error, lr=0.1):
        delta = self.df(self.z) * error
        gW = numpy.atleast_2d(delta).dot(numpy.atleast_2d(self.x.T))
        self.W -= lr * gW
        if not self.nobias:
            self.b -= lr * numpy.sum(delta)
        return self.W.T.dot(delta)

class NN(object):
    def __init__(self, list_of_num_node, list_of_activation_function):
        self.Ls = [Layer(list_of_num_node[i], list_of_num_node[i+1],
                         list_of_activation_function[i][0], list_of_activation_function[i][1],
                         True) for i in xrange(len(list_of_num_node) - 1)]

    def forward(self, x):
        z = x
        for l in self.Ls:
            z = l.forward(z)
        return z

    def backward(self, error):
        delta = error
        for i in xrange(len(self.Ls) - 1, -1, -1):
            l = self.Ls[i]
            delta = l.backward(delta)

if __name__ == "__main__":
    nn = NN([2, 5, 1], [(sigmoid, dsigmoid), (sigmoid, dsigmoid)])
    
    X = numpy.array([[0, 0], [0 ,1], [1, 0], [1, 1]])
    T = numpy.array([0, 1, 1, 0])

    epoches = 10000
    for epoch in xrange(epoches):
        cost = 0
        for i in numpy.random.permutation(len(X)):
            x, t = numpy.atleast_2d(X[i]).T, T[i]
            y = nn.forward(x)
            nn.backward(y - t)
            cost += (y - t)**2
        if epoch % 1000 == 0:
            print "cost = ", 0.5 * cost / len(X)

    for x in X:
        x = numpy.atleast_2d(x).T
        print "%s : predicted %s" % (x, nn.forward(x))
