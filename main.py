import numpy as np
import itertools as it
from collections import deque, Counter
import regex as re
import sys
import unicodedata as ud
import nltk


def cos(u, v):
    a = np.linalg.norm(u, 2)
    b = np.linalg.norm(v, 2)
    return 1 - np.dot(u, v) / a / b


def pr(A, max_iters=None, k=1e-3, d=0.15):
    divisors = A.sum(axis=1, keepdims=True)
    divisors[divisors == 0] = 1
    i = 0
    A_hat = A / divisors
    n = A_hat.shape[1]
    v = np.random.rand(n, 1)
    v /= np.linalg.norm(v, 2)
    u = None
    A_hat *= (1-d)
    A_hat += d / n
    while u is None or np.sum(np.square(u-v)) >= k:
        u = v
        v = A_hat.dot(v)
        i += 1
        if max_iters is not None and i == max_iters:
            break
    return v.reshape(v.shape[0])


def tokenize(istrm):
    return (t for t in re.split(r'[\s\p{Mn}\p{P}]+', istrm)
            if t != '')


def ngrams(tokens, n=3):
    ctx = deque((), maxlen=n)
    ctx += it.islice(tokens, n)
    if len(ctx) > 0:
        yield tuple(ctx)
    for t in tokens:
        ctx.append(t)
        yield tuple(ctx)


def recursive_ngrams(tokens, n=3):
    for ctx in ngrams(tokens, n=n):
        yield ctx
        for k in range(1, n):
            yield from ngrams(iter(ctx), n=k)


def pmi(tokens, n=5):
    tokens, ttokens = it.tee(tokens)
    F = Counter(tokens)
    v = len(F)
    A = np.zeros((v, v))
    T = {t: i for i, t in enumerate(F.keys())}
    for ctx in ngrams(ttokens, n=n):
        for t in ctx:
            for w in ctx:
                A[T[t], T[w]] += 1 / n
    A = np.log(A + 1)
    for t in F:
        for w in F:
            A[T[t], T[w]] -= np.log(F[t] * F[w])
    return (T, A)


def tf_idf(tokens, n=32):
    tf = Counter()
    df = Counter()
    d = 0
    queue = deque(maxlen=n)
    while True:
        ctx = tuple(it.islice(tokens, n))
        queue += ctx
        if len(ctx) == 0:
            break
        d += 1
        touch = set()
        for t in ctx:
            tf[t] += 1
            touch.add(t)
        for t in touch:
            df[t] += 1
    T = tuple(tf.keys())
    R = np.array([tf[t] * np.log(d / df[t]) for t in T])
    R /= np.linalg.norm(R, 2)
    return dict(zip(T, R))


class Traker(object):
    def __init__(self, tokens, n=5, dim=None):
        tokens, ttokens = it.tee(tokens)
        self.tf_idf = tf_idf(ttokens)
        tokens, ttokens = it.tee(tokens)
        T, J = pmi(tokens, n=n)
        R = pr(J)
        self.vocab = T
        R /= np.linalg.norm(R, 2)
        self.tr = {t: R[i] for t, i in T.items()}

    def stoprank(self, tokens):
        if isinstance(tokens, str):
            tokens = (tokens,)
        v = [0.7*(self.tr[t]) + 0.3*self.tf_idf[t] for t in tokens]
        m = np.mean(v)
        return m

    def keygrams(self, tokens, n=3):
        candidates = recursive_ngrams(tokens, n=n)
        C = Counter(map(tuple, candidates))
        R = {k: self.stoprank(k) / np.log(f+1) / (len(k)+1)
             for k, f in C.items()}
        V = list(C.keys())
        V.sort(key=lambda k: R[k])
        return V

if __name__ == '__main__':
    np.seterr(all='raise')
    tokens = tuple(tokenize(sys.stdin.read().lower()))
    # strip away conjunctions and determinants ― anything that isn't useful for
    # consideration as part of a key -word or -phrase
    # forbidden_tags = {'CC', 'DT', 'PRP', 'VBZ', 'IN', 'TO', 'VBP', 'MD'}
    # tokens = tuple(t for t, tag in nltk.pos_tag(tokens)
    #                if tag not in forbidden_tags)
    # ...
    # or just use a list
    stops = set(nltk.corpus.stopwords.words('english'))
    stops |= {'ur', 'u', 'r'}
    tokens = tuple(t for t in tokens if t not in stops)
    R = Traker(tokens, n=5, dim=64)
    print(R.keygrams(tokens)[:10])
