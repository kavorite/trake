import numpy as np
import itertools as it
from collections import deque, Counter
import regex as re
import sys
import unicodedata as ud
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats


def nrm2(v):
    return np.sqrt(np.sum(np.square(v)))


def cos(u, v):
    a, b = map(nrm2, (u, v))
    cosine = np.clip(np.dot(u, v) / a / b, -1, 1)
    return 1 - np.arccos(cosine) / np.pi


def pr(A, max_iters=None, k=1e-3, d=0.15):
    divisors = A.sum(axis=1, keepdims=True)
    divisors[divisors == 0] = 1
    i = 0
    A /= divisors
    n = A.shape[1]
    v = np.random.rand(n, 1)
    v *= (1 / nrm2(v))
    u = None
    A_hat = A*(1-d) + (d/n)
    while u is None or np.sum(np.square(u-v)) >= k:
        u = v
        v = A_hat.dot(v)
        i += 1
        if max_iters is not None and i == max_iters:
            break
    return v.reshape(v.shape[0])


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
            ctx, tctx = it.tee(ctx)
            yield from ngrams(tctx, n=k)


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


def tf_idf(tokens, n=5):
    tf = Counter()
    df = Counter()
    d = 0
    for ctx in ngrams(tokens, n=n):
        d += 1
        touch = set()
        for t in ctx:
            tf[t] += 1/n
            touch.add(t)
        for t in touch:
            df[t] += 1
    return {t: tf[t] * np.log(d / df[t]) for t in tf.keys()}


class MoodyEmbeddings(object):
    def __init__(self, tokens, n=5, dim=None):
        tokens, ttokens = it.tee(tokens)
        self.tf_idf = tf_idf(ttokens, n=n)
        tokens, ttokens = it.tee(tokens)
        T, J = pmi(tokens, n=n)
        U, S, V_h = np.linalg.svd(J, full_matrices=False)
        dim = 64 if dim is None else dim
        dim = min(dim, V_h.shape[0])

        V = V_h[:, :dim]
        A = J.dot(V)
        A = np.linalg.lstsq(A, V, rcond=None)[0]

        R = np.array([[cos(V[i], V[j]) for t, i in T.items()]
                      for j in range(len(T))])
        R = pr(R)

        self.vocab = {t: V[i] for t, i in T.items()}
        self.induction_matrix = A
        self.dim = V.shape[1]
        self.tr = {t: R[i] for t, i in T.items()}

    def embed(self, token, placeholder=None):
        if placeholder is None:
            placeholder = np.zeros(self.dim)
        return self.vocab[token] if token in self.vocab else placeholder

    def oneshot(self, tokens):
        M = np.array([self.embed(t) for t in tokens])
        v = np.mean(M, axis=0)
        v = v.reshape(v.shape[0])
        return self.induction_matrix.dot(v)

    def stoprank(self, tokens):
        if isinstance(tokens, str):
            tokens = (tokens,)
        # M = np.array([self.embed(t) for t in tokens])
        # u = np.mean(M, axis=0).reshape(M.shape[1])
        # v = self.induction_matrix.dot(u)
        # v = [self.tf_idf[t] for t in tokens]
        v = [1-self.tr[t] for t in tokens]
        m = np.mean(v)
        return m

    def stops(self, alpha=0.95):
        R = {t: self.stoprank(t) for t in self.vocab
             for t in self.vocab.keys()}
        # R = {t: self.tf[t] for t in self.vocab.keys()}
        parameters = stats.expon.fit([tuple(R.values())])
        exp = dict(zip(('loc', 'scale'), parameters))
        alpha = stats.expon.ppf(alpha, **exp)
        S = list(t for t, x in R.items() if alpha < x)
        S.sort(key=lambda t: R[t])
        return S

    def summary_candidates(self, tokens, alpha=0.99, n=2):
        stops = set(self.stops(alpha=alpha))
        queue = []
        C = Counter()
        for t in tokens:
            if t in stops and len(queue) > 0:
                for ctx in ngrams(queue, n=n):
                    ctx = tuple(ctx)
                    C[ctx] += 1
                queue = []
            else:
                queue.append(t)
        V = list(C.keys())
        V.sort(key=lambda ctx: self.stoprank(ctx) / np.log(C[ctx]+1), reverse=True)
        return V

    def summarize(self, tokens, embed=None, n=5):
        if embed is None:
            embed = self.embed
        K = dict(())
        for k in recursive_ngrams(tokens, n=n):
            K[k] = len(K)
        V = tuple(map(self.oneshot, K))
        d = len(V)
        A = np.array(tuple(cos(u, v) for v in V for u in V)).reshape((d, d))
        R = pr(A)
        R = list(zip(K, R))
        R.sort(lambda pair: pair[1], reverse=True)
        return OrderedDict(R)


def tr(tokens, max_iters=None, n=2, d=0.15):
    tokens = tuple(tokens)
    T, V = pmi(iter(tokens), n=n)
    K = {}
    for ctx in recursive_ngrams(tokens, n=n):
        ctx = tuple(ctx)
        if len(ctx) > 1 and ctx not in K:
            K[ctx] = len(T) + len(K)
    v = len(T) + len(K)
    A = np.zeros((v, v))
    for t, i in T.items():
        for w, j in T.items():
            A[i, j] = V[i, j]
    for ctx_a, i in K.items():
        for ctx_b, j in K.items():
            keypairs = tuple(zip(ctx_a, ctx_b))
            A[i, j] = sum(V[T[t], T[w]] for t, w in keypairs) / len(keypairs)
    R = pr(A, max_iters=max_iters, d=d)
    K.update(T)
    return {t: R[i] for t, i in K.items()}


def strip_punct(s):
    ledger = dict.fromkeys(i for i in range(sys.maxunicode)
                           if ud.category(chr(i)).startswith('P'))
    return s.translate(ledger)


def tokenize(istrm):
    return re.split(r'[\s\p{Mn}\p{P}0-9]+', istrm)


np.seterr(all='raise')
tokens = tuple(tokenize(sys.stdin.read().lower()))
M = MoodyEmbeddings(tokens, n=5, dim=64)
print(M.summary_candidates(tokens)[:10])

# print(list(M.stoprank().keys()).index('textrank') / len(M.vocab))
# print(M.vocab['the'])
# print(M.vocab['and'])
# print(M.vocab['textrank'])
# print(M.vocab['unsupervised'])
# print(M.vocab['unigrams'])
# R = M.tr(tokens, n=5)
# print(dict(R.items()[:10]))
# v = M.oneshot(tokens)
# R = {t: nrm2(cos(u, v)) for t, u in M.vocab.items()}
# R = tr(iter(tokens), n=1)
# K = list(R.keys())
# K.sort(key=lambda t: R[t], reverse=True)
# K = K[:10]
# print(K)
