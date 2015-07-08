import sys
import collections

import numpy
from sklearn.datasets import load_svmlight_file

class DSSM(object):
    def __init__(self, num_input, num_hidden, num_output,
                 activation=None, rng=None):
        self.activation = activation
        if rng is None:
            rng = numpy.random.RandomState(1234)
        w = numpy.sqrt(6. / (num_input + num_hidden))
        self.W2 = rng.normal(
            -w, w,
            (num_hidden, num_input)).astype(numpy.float32)
        self.b2 = numpy.ones(num_hidden, dtype=numpy.float32)
        w = numpy.sqrt(6. / (num_hidden + num_output))
        self.W3 = rng.normal(
            -w, w,
            (num_output, num_hidden)).astype(numpy.float32)
        self.b3 = numpy.ones(num_output, dtype=numpy.float32)

    def _cosine(self, q, d):
        ip = q.dot(d)
        return  ip / numpy.sqrt(q.dot(q.T)) / numpy.sqrt(d.dot(d.T)) if ip != 0 else 0

    def _calc_abc(self, yq, yd):
        b = yq.dot(yq)
        b = 1.0 / numpy.sqrt(b) if b != 0 else 0
        c = yd.dot(yd)
        c = 1.0 / numpy.sqrt(c) if c != 0 else 0
        return yq.dot(yd), b, c

    def _calc_delta(self, lq, ld, error1, error2):
        dlq = (1 - lq) * (1 + lq) * error1
        dld = (1 - ld) * (1 + ld) * error2
        return dlq, dld

    def predict(self, q, d):
        lq = self.activation(q.dot(self.W2.T) + self.b2)
        yq = self.activation(lq.dot(self.W3.T) + self.b3)
        ld = self.activation(d.dot(self.W2.T) + self.b2)
        yd = self.activation(ld.dot(self.W3.T) + self.b3)
        return self._cosine(yq, yd)

    def fit(self, q, d, nds, gamma=0.1, lr=0.01):
        l2q = self.activation(q.dot(self.W2.T) + self.b2)
        yq = self.activation(l2q.dot(self.W3.T) + self.b3)
        l2d = self.activation(d.dot(self.W2.T) + self.b2)
        yd = self.activation(l2d.dot(self.W3.T) + self.b3)

        a, b, c = self._calc_abc(yq, yd)
        rp = a * b * c
        dlq, dld = self._calc_delta(yq, yd,
                         b * c * yd - a * c * b * b * b * yq,
                         b * c * yq - a * b * c * c * c * yd)
        gW3p = numpy.atleast_2d(dlq).T.dot(numpy.atleast_2d(l2q)) + \
            numpy.atleast_2d(dld).T.dot(numpy.atleast_2d(l2d))

        gb3p = (dlq + dld) * self.b3
        dlq, dld = self._calc_delta(l2q, l2d,
                                    dlq.dot(self.W3), dld.dot(self.W3))
        gW2p = numpy.atleast_2d(dlq).T.dot(numpy.atleast_2d(q)) + \
            numpy.atleast_2d(dld).T.dot(numpy.atleast_2d(d))
        gb2p = (dlq + dld) * self.b2

        alpha = numpy.zeros(nds.shape[0])
        gW2 = numpy.zeros_like(self.W2)
        gW3 = numpy.zeros_like(self.W3)
        gb2 = numpy.zeros_like(self.b2)
        gb3 = numpy.zeros_like(self.b3)

        for i in xrange(nds.shape[0]):
            d = nds[i]
            l2d = self.activation(d.dot(self.W2.T) + self.b2)
            yd = self.activation(l2d.dot(self.W3.T) + self.b3)
            a, b, c = self._calc_abc(yq, yd)
            rn = a * b * c
            dlq, dld = self._calc_delta(yq, yd,
                                        b * c * yd - a * c * b * b * b * yq,
                                        b * c * yq - a * b * c * c * c * yd)
            gW3n = numpy.atleast_2d(dlq).T.dot(numpy.atleast_2d(l2q)) + \
                numpy.atleast_2d(dld).T.dot(numpy.atleast_2d(l2d))
            gb3n = (dlq + dld) * self.b3
            dlq, dld = self._calc_delta(l2q, l2d,
                                        dlq.dot(self.W3), dld.dot(self.W3))
            gW2n = numpy.atleast_2d(dlq).T.dot(numpy.atleast_2d(q)) + \
                numpy.atleast_2d(dld).T.dot(numpy.atleast_2d(d))
            gb2n = (dlq + dld) * self.b2

            alpha[i] = numpy.exp(-gamma * (rp - rn))
            gW2 += alpha[i] * (gW2p - gW2n)
            gW3 += alpha[i] * (gW3p - gW3n)
            gb2 += alpha[i] * (gb2p - gb2n)
            gb3 += alpha[i] * (gb3p - gb3n)
            
        sum_of_alpha = numpy.sum(alpha)
        self.W2 -= -gamma * lr / (1. + sum_of_alpha) * gW2
        self.W3 -= -gamma * lr / (1. + sum_of_alpha) * gW3
        self.b2 -= -gamma * lr / (1. + sum_of_alpha) * gb2
        self.b3 -= -gamma * lr / (1. + sum_of_alpha) * gb3
        
def main():
    Q, _ = load_svmlight_file("query.svmdata", n_features=2**15)
    D, _ = load_svmlight_file("doc.svmdata", n_features=2**15)

    R = []
    Ridx = collections.defaultdict(set)

    with open("cranqrel") as f:
        for line in f:
            a = line.strip().split(" ")
            try:
                idq = int(a[0]) - 1
                idd = int(a[1]) - 1
                rel = int(a[2])
                if rel == 5:
                    rel = -1
                if rel == -1:
                    continue
                R.append(tuple([idq, idd, rel]))
                Ridx[idq].add(idd)
            except Exception as e:
                print >> sys.stderr, e

    dssm = DSSM(2**15, 10, 6, numpy.tanh)
    epoches = 10
    for epoch in xrange(epoches):
        cost = 0
        for x in R:
            q = Q[x[0]].todense().A1
            d = D[x[1]].todense().A1
            idx = list(set(numpy.random.randint(0, D.shape[0], 20).tolist()) - Ridx[x[0]])
            idx = numpy.random.permutation(idx)
            nd = numpy.asarray(D[idx[0:4]].todense())
            dssm.fit(q, d, nd)
            cost += (dssm.predict(q, d) - 1)**2
            for dd in nd:
                cost += dssm.predict(q, dd)**2
        print "cost = ", cost / 2 / (len(R) * 5)
            
if __name__ == "__main__":
    main()

