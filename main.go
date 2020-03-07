package main

import (
    "math"
    "time"
    "math/rand"
    "regexp"
    "strings"
    "sort"
    "io/ioutil"
    "os"
    "fmt"

    "gonum.org/v1/gonum/mat"
)

func init() {
    rand.Seed(int64(time.Now().Nanosecond()))
}

func Cos(u, v mat.Vector) float64 {
    return 1 - mat.Dot(u, v)/mat.Norm(u, 2)/mat.Norm(v, 2)
}

func PR(A *mat.Dense, qerr, d float64) mat.Vector {
    tmp := mat.DenseCopyOf(A)
    return PRInPlace(tmp, qerr, d)
}

func PRInPlace(A *mat.Dense, e, d float64) *mat.VecDense {
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
    for i := 0; i < k; i++ {
        v := A.RowView(i)
        n := mat.Sum(v)
        if n == 0 {
            n = 1
        } else {
            n = 1/n
        }
        data := A.RawRowView(i)
        for j := 0; j < v.Len(); j++ {
            data[i] *= n
        }
    }
    n := float64(k)
    A.Apply(func(i, j int, x float64) float64 {
        return x * (1-d) + (d/n)
    }, A)
    v := mat.NewDense(k, 1, make([]float64, k))
    v.Apply(func(i, j int, x float64) float64 { return rand.Float64()
    }, v)
    v.Scale(1/mat.Norm(v, 2), v)
    for {
        u := mat.DenseCopyOf(v)
        v.Mul(A, v)
        u.Sub(u, v)
        u.Apply(func(i, j int, x float64) float64 {
            return x * x
        }, u)
        qerr := mat.Sum(u)
        if qerr < e {
            break
        }
        // fmt.Println(qerr)
    }
    v.Scale(1/mat.Sum(v), v)
    return mat.VecDenseCopyOf(v.ColView(0))
}

type Tokens []string

func Tokenize(raw string) (T Tokens) {
    delim := regexp.MustCompile(`[\s\p{Mn}\p{P}]+`)
    T = Tokens(delim.Split(raw, -1))
    return
}

func (T Tokens) String() string {
    return strings.Join([]string(T), " ")
}

func (T Tokens) Chunks(n int, f func(Tokens)) {
    i := 0
    for i <= len(T)-n {
        k := i+n
        if k > len(T) {
            k = len(T)
        }
        f(T[i:k])
        i = k
    }
}
func (T Tokens) NGrams(n int, f func(Tokens)) {
    i := 0
    for i <= len(T)-n {
        f(T[i:i+n])
        i++
    }
}

func (T Tokens) RecNGrams(n int, f func(Tokens)) {
    T.NGrams(n, func(S Tokens) {
        f(S)
        for k := 1; k < n-1; k++ {
            S.NGrams(k, f)
        }
    })
}

func (T Tokens) Vocab() (V map[string]int) {
    V = make(map[string]int, 256)
    for _, t := range T {
        if _, ok := V[t]; !ok {
            V[t] = len(V)
        }
    }
    return
}

type Doc struct {
    Tokens
    Vocab map[string]int
}

func DocFrom(T Tokens) Doc {
    V := T.Vocab()
    return Doc{T, V}
}

func (D Doc) V() int {
    return len(D.Vocab)
}

func (D Doc) TF() (tf []float64) {
    tf = make([]float64, D.V())
    for _, t := range D.Tokens {
        tf[D.Vocab[t]] += 1
    }
    return
}

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
    for i, x := range idf {
        idf[i] = math.Log(float64(d) / x)
    }
    return
}

func (D Doc) PMI(n int, tf []float64) (A *mat.Dense) {
    v := D.V()
    A = mat.NewDense(v, v, make([]float64, v*v))
    D.NGrams(n, func(S Tokens) {
        j := 1 / float64(n)
        for _, t := range S {
            i := D.Vocab[t]
            for _, w := range S {
                k := D.Vocab[w]
                A.Set(i, k, A.At(i, k) + j)
            }
        }
    })
    A.Apply(func(i, j int, x float64) float64 {
        return math.Log((x+1)/tf[i]/tf[j])
    }, A)
    return
}

type Traker struct {
    Doc
    TF, IDF, TR *mat.VecDense
}

func TrakeFrom(D Doc, s, j int) Traker {
    tf := D.TF()
    A := D.PMI(j, tf)
    tr := PRInPlace(A, 1e-3, 0.15)
    fmt.Println("safe")
    R := Traker {
        Doc: D,
        TF: mat.NewVecDense(D.V(), tf),
        IDF: mat.NewVecDense(D.V(), D.IDF(s)),
        TR: tr,
    }
    return R
}

func (R Traker) Score(S Tokens) (x float64) {
    for _, t := range S {
        i := R.Vocab[t]
        x += 0.3*R.TF.AtVec(i)*R.IDF.AtVec(i) + 0.7*R.TR.AtVec(i)
    }
    x /= float64(len(S))
    return
}

type Keygrams struct {
    Candidates []Tokens
    Ranks []float64
}

func (K Keygrams) Len() int {
    return len(K.Candidates)
}

func (K Keygrams) Less(i, j int) bool {
    return K.Ranks[i] < K.Ranks[j]
}

func (K Keygrams) Swap(i, j int) {
    u, v := K.Candidates, K.Ranks
    u[i], u[j] = u[j], u[i]
    v[i], v[j] = v[j], v[i]
}

func (R Traker) Keygrams(n int) (K Keygrams) {
    K.Candidates = make([]Tokens, 1024)
    R.RecNGrams(n, func(S Tokens) {
        K.Candidates = append(K.Candidates, S)
    })
    K.Ranks = make([]float64, len(K.Candidates))
    for i, S := range K.Candidates {
        K.Ranks[i] = R.Score(S)
    }
    return
}

func (K Keygrams) Sort() {
    sort.Sort(K)
}

func main() {
    buf, err := ioutil.ReadAll(os.Stdin)
    if err != nil {
        panic(err)
    }
    T := Tokenize(string(buf))
    D := DocFrom(T)
    R := TrakeFrom(D, 32, 5)
    K := R.Keygrams(3)
    K.Sort()
    fmt.Println(K.Candidates[:10])
}
