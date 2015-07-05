import sys
import cPickle as pickle
import numpy

def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))

class CBOW(object):
    def __init__(self, dim, n_vocab, path):
        self.n_vocab = n_vocab
        self.path = path
        self.syn0 = numpy.random.uniform(
            -0.5/dim, 0.5/dim, (n_vocab, dim)).astype(numpy.float32)
        self.syn1 = numpy.zeros((n_vocab, dim))

    def preprocess(self):
        vocab_hash = {}
        vocab_items = []
        vocab_count = {}

        for token in ["bol", "eol"]:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(token)
            vocab_count[vocab_hash[token]] = 0

        # token -> index
        word_count = 0
        with open(self.path) as f:
            for line in f:
                for token in line.split():
                    if token not in vocab_hash:
                        vocab_hash[token] = len(vocab_items)
                        vocab_items.append(token)
                        vocab_count[vocab_hash[token]] = 0
                    vocab_count[vocab_hash[token]] += 1
                    word_count += 1
                vocab_count[vocab_hash["bol"]] += 1
                vocab_count[vocab_hash["bol"]] += 1
                word_count += 2

        vocab_hash.clear()
        for i, pair in enumerate(sorted(vocab_count.items(), key=lambda x:x[1], reverse=True)[:self.n_vocab-1]):
            idx, count = pair
            vocab_hash[vocab_items[idx]] = i
        vocab_hash["unk"] = len(vocab_hash)
        self.vocab_hash = vocab_hash
        
    def negative_sampling(self, count):
        return numpy.random.randint(low=0, high=self.n_vocab, size=count)

    def train(self, window=5, neg=20, alpha=0.1):
        word_count = 0
        with open(self.path) as f:
            for line in f:
                sent = ["bol"] + line.split() + ["eol"]
                sent = [self.vocab_hash[token] if token in self.vocab_hash else self.vocab_hash["unk"] for token in sent]
                for pos, token in enumerate(sent):
                    if word_count % 1000 == 0:
                        print word_count
                    cs = max(0, pos - window)
                    ce = min(len(sent), pos + window)
                    context = sent[cs:pos] + sent[pos+1:ce]
                    neu1 = numpy.mean([self.syn0[c] for c in context], axis=0)
                    neu1e = numpy.empty_like(neu1)

                    # negative sampling
                    X = numpy.array([self.syn1[token]] + \
                                    [self.syn1[token] for token in self.negative_sampling(neg)])
                    T = numpy.array([1] + [0 for i in xrange(neg)])
                    g = (T - sigmoid(neu1.dot(X.T))) * alpha
                    # neu1e = numpy.sum(g * X.T, axis=0).T
                    # self.syn1[token] += numpy.sum(g * (neu1 * numpy.ones(X.shape)).T, axis=0).T
                    
                    pairs = [(token, 1)] + \
                        [(token, 0) for token in self.negative_sampling(neg)]
                    for token, label in pairs:
                        z = neu1.dot(self.syn1[token])
                        f = sigmoid(z)
                        g = (label - f) * alpha
                        neu1e += g * self.syn1[token]
                        self.syn1[token] += g * neu1

                    for c in context:
                        self.syn0[c] += neu1e

                    word_count += 1

def cbow_train():
    cbow = CBOW(100, 50000, "text8_1000")
    cbow.preprocess()
    cbow.train(5, 20, 0.025)
    with open("cbow_test", "wb") as f:
        pickle.dump(cbow, f)

def cbow_vocab():
    with open("cbow_test", "rb") as f:
        cbow = pickle.load(f)
        # print cbow.vocab_hash.keys()
             
if __name__ == "__main__":
    #cbow_train()
    cbow_vocab()
