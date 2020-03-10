package trake

import (
    "math"
    "math/rand"
    "regexp"
    "sort"
    "strings"
    "time"

    "github.com/james-bowman/sparse"
    "gonum.org/v1/gonum/mat"
)

func init() {
    rand.Seed(int64(time.Now().Nanosecond()))
}

// PR stores a sparse clone of the matrix A, L1-normalizes its rows, and
// calculates the PageRank of the elements, then returns the result as a dense
// vector with the given quadratic error e and damping factor d.
func PR(A mat.Matrix, e, d float64) *mat.VecDense {
    if d < 0 || d > 1 {
        d = 0.15
    }
    if e < 0 || e > 1 {
        e = 1e-3
    }
    k, rowc := A.Dims()
    if rowc != k {
        panic("dimension mismatch")
    }
    A_hat := sparse.NewCSR(k, k, make([]int, 0, k), make([]int, 0, k), make([]float64, 0, k))
    A_hat.Clone(A)
    for i := 0; i < k; i++ {
        v := A_hat.RowView(i)
        n := mat.Sum(v)
        if n == 0 {
            n = 1
        } else {
            n = 1 / n
        }
        for j := 0; j < v.Len(); j++ {
            A_hat.Set(i, j, v.AtVec(j)*n)
        }
    }
    n := float64(k)
    A_hat.DoNonZero(func(i, j int, x float64) {
        x *= (1 - d)
        x += d / n
        A_hat.Set(i, j, x)
    })
    v := mat.NewDense(k, 1, make([]float64, k))
    v.Apply(func(i, j int, x float64) float64 {
        return rand.Float64() / n
    }, v)
    for {
        u := mat.DenseCopyOf(v)
        v.Mul(A_hat, v)
        u.Sub(u, v)
        u.MulElem(u, u)
        qerr := mat.Sum(u)
        if qerr < e {
            break
        }
    }
    v.Scale(1/mat.Sum(v), v)
    return mat.VecDenseCopyOf(v.ColView(0))
}

// Tokens is a type alias for lexically manipulating a tokenized input.
type Tokens []string

// Tokenize tokenizes the given src and returns it as a string slice for
// lexical manipulations, ignoring all tokens t ∈ stops.
func Tokenize(src string, stops []string) (T Tokens) {
    stops = append(stops, "")
    S := make(map[string]struct{}, len(stops))
    for _, t := range stops {
        S[t] = struct{}{}
    }
    delim := regexp.MustCompile(`[\s\p{Mn}\p{P}0-9]+`)
    T = delim.Split(src, -1)
    i := 0
    for _, t := range T {
        // t = delim.ReplaceAllString(t, "")
        if _, ok := S[t]; ok {
            continue
        }
        T[i] = t
        i++
    }
    T = T[:i]
    return
}

// String gives a string representation of a series of tokens, as a
// scheme-case hashtag.
func (T Tokens) String() string {
    return "#" + strings.Join([]string(T), "-")
}

// Chunks subdivides the document into n-grams, yielding each in turn. Unlike
// Tokens.NGrams, the arguments passed to f never overlap.
func (T Tokens) Chunks(n int, f func(Tokens)) {
    i := 0
    for i <= len(T)-n {
        k := i + n
        if k > len(T) {
            k = len(T)
        }
        f(T[i:k])
        i = k
    }
}

// NGrams subdivides a document into successive 'windows' of overlapping tokens
// and calls f on each sequence of lexical units in turn.
func (T Tokens) NGrams(n int, f func(Tokens)) {
    i := 0
    for i <= len(T)-n {
        f(T[i : i+n])
        i++
    }
}

// RecNGrams calls NGrams recursively, such that all k-grams of the input
// from 1 to n are passed to f.
func (T Tokens) RecNGrams(n int, f func(Tokens)) {
    T.NGrams(n, func(S Tokens) {
        f(S)
        for k := 1; k < n-1; k++ {
            S.NGrams(k, f)
        }
    })
}

// Vocab returns a dictionary mapping each token to an index corresponding to
// the order in which they appear.
func (T Tokens) Vocab() (V map[string]int) {
    V = make(map[string]int, 256)
    for _, t := range T {
        if _, ok := V[t]; !ok {
            V[t] = len(V)
        }
    }
    return
}

// Doc is a Tokens sequence associated with a precomputed vocabulary
// mapping Vocab (see Tokens.Vocab).
type Doc struct {
    Tokens
    Vocab map[string]int
}

// DocFrom constructs a Doc from the given Tokens.
func DocFrom(T Tokens) Doc {
    V := T.Vocab()
    return Doc{T, V}
}

// V returns the size of the Doc's Vocab.
func (D Doc) V() int {
    return len(D.Vocab)
}

// TF returns the L1-normalized term frequencies of each token that occurs in
// D.
func (D Doc) TF() (tf []float64) {
    tf = make([]float64, D.V())
    sigma := float64(0)
    for _, t := range D.Tokens {
        tf[D.Vocab[t]] += 1
        sigma += 1
    }
    v := mat.NewVecDense(D.V(), tf)
    v.ScaleVec(1/sigma, v)
    return
}

// IDF returns the L1-normalized inverse document frequency of the terms in the
// document such that idf[D.Vocab[t]] = log(|D| / |{d ∈ D; t ∈ d|) for the
// tokenized contents of the document D as subdivided into a sequence of
// s-grams, d.
func (D Doc) IDF(s int) (idf []float64) {
    d := 0
    idf = make([]float64, D.V())
    D.Tokens.Chunks(s, func(S Tokens) {
        d++
        touch := make(map[string]struct{}, d)
        for _, t := range S {
            touch[t] = struct{}{}
        }
        for t := range touch {
            idf[D.Vocab[t]] += 1
        }
    })
    sigma := float64(0)
    for i, x := range idf {
        x = math.Log(float64(d) / x)
        idf[i] = x
        sigma += x
    }
    v := mat.NewVecDense(D.V(), idf)
    v.ScaleVec(1/sigma, v)
    return
}

// PMI returns an adjacency matrix of terms' cooccurrence frequencies,
// downscaled by each terms' probability of occurring (term frequency), such
// that A[D.Vocab[t], D.Vocab[w]] is the adjacency of tokens t and w. The
// first argument controls the cutoff for nonzero sparse elements, the second
// controls the size of context used to learn cooccurrences, and the last is
// L1-normalized term frequencies (see Tokens.TF).
func (D Doc) PMI(c, n int, tf []float64) (A *sparse.DOK) {
    v := D.V()
    A = sparse.NewDOK(v, v)
    D.NGrams(n, func(S Tokens) {
        x := 1 / float64(n)
        for _, t := range S {
            i := D.Vocab[t]
            for _, w := range S {
                j := D.Vocab[w]
                A.Set(i, j, A.At(i, j)+x)
            }
        }
    })
    if c < 0 {
        return
    }
    X := make([]float64, 0, A.NNZ())
    A.DoNonZero(func(i, j int, x float64) {
        x = math.Log((x + 1) / tf[i] / tf[j])
        A.Set(i, j, x)
        k := sort.SearchFloat64s(X, -x)
        X = append(X[:k], append([]float64{x}, X[k:]...)...)
    })
    if c > len(X)-1 {
        c = len(X) - 1
    }
    min := X[c]
    J := sparse.NewDOK(v, v)
    A.DoNonZero(func(i, j int, x float64) {
        if x >= min {
            J.Set(i, j, x)
        }
    })
    A = J
    return
}

// Traker is a Doc with a precomputed TF, IDF, and TextRank for each token
// appearing in Doc.Vocab.
type Traker struct {
    Doc
    TF, IDF, TR []float64
}

type OptKind int

const (
    KPmiWindowSize OptKind = 0
    KPmiNnz        OptKind = iota
    KIdfChunkSize  OptKind = iota
)

type Opt struct {
    Kind  OptKind
    Value int
}

func OPmiNnz(n int) Opt {
    return Opt{KPmiNnz, n}
}

func OPmiWindowSize(n int) Opt {
    return Opt{KPmiWindowSize, n}
}

func OIdfChunkSize(n int) Opt {
    return Opt{KIdfChunkSize, n}
}

// From returns a new Traker from the given Doc, using the given options.
func From(D Doc, opts ...Opt) Traker {
    c, s, j := -1, 32, 5
    tf := D.TF()
    for _, opt := range opts {
        switch opt.Kind {
        case KPmiNnz:
            c = opt.Value
            if c < 0 {
                c = int(math.Round(math.Log2(float64(D.V())))) + 1
            }
        case KPmiWindowSize:
            j = opt.Value
        case KIdfChunkSize:
            s = opt.Value
        default:
            panic("unrecognized option")
        }
    }
    A := D.PMI(c, j, tf)
    R := Traker{
        Doc: D,
        TF:  tf,
        IDF: D.IDF(s),
        TR:  PR(A, 1e-3, 0.3).RawVector().Data,
    }
    return R
}

// Score returns an aggregate of R's various metrics to rank the
// significance of a given subsequence of R.Tokens to the overall subject
// matter of the document.
func (R Traker) Score(S Tokens) (x float64) {
    for _, t := range S {
        i := R.Vocab[t]
        x += 0.3 * R.IDF[i] * R.TF[i]
        x += 0.7 * R.TR[i]
    }
    x /= float64(len(S))
    return
}

// Keygrams is a series of candidate phrases and their corresponding ranks,
// implementing sort.Interface.
type Keygrams struct {
    Candidates []Tokens
    Ranks      []float64
    Cap        int
}

func (K *Keygrams) Len() int {
    return len(K.Candidates)
}

func (K *Keygrams) Less(i, j int) bool {
    return K.Ranks[i] < K.Ranks[j]
}

func (K *Keygrams) Swap(i, j int) {
    u, v := K.Candidates, K.Ranks
    u[i], u[j] = u[j], u[i]
    v[i], v[j] = v[j], v[i]
}

// Insert performs binary search to insert the given candidate S such that
// elements remain sorted by their respective ranks in ascending order.
func (K *Keygrams) Insert(S Tokens, x float64) {
    i := sort.SearchFloat64s(K.Ranks, x)
    K.Candidates = append(K.Candidates, nil)
    copy(K.Candidates[i+1:], K.Candidates[i:])
    K.Candidates[i] = S

    K.Ranks = append(K.Ranks, 0)
    copy(K.Ranks[i+1:], K.Ranks[i:])
    K.Ranks[i] = x
}

func (K *Keygrams) Sort() {
    sort.Sort(K)
}

// Keygrams catalogs and returns Keygrams using the given Traker.
func (R Traker) Keygrams(n, c int) (K *Keygrams) {
    F := make(map[string]float64, c)
    C := make(map[string]struct{}, c)
    K = &Keygrams{
        Candidates: make([]Tokens, 0, c),
        Ranks:      make([]float64, 0, c),
        Cap:        c,
    }
    R.RecNGrams(n, func(S Tokens) {
        k := S.String()
        if _, ok := F[k]; !ok {
            F[k] = 0
        }
        F[k] += 1
        if _, ok := C[k]; !ok {
            C[k] = struct{}{}
            x := R.Score(S) / math.Log(F[k]+1) / float64(len(S)+1)
            K.Insert(S, x)
        }
        if len(K.Candidates) > c {
            for _, D := range K.Candidates[c:] {
                delete(C, D.String())
            }
            K.Candidates = K.Candidates[:c]
            K.Ranks = K.Ranks[:c]
        }
    })
    return
}
