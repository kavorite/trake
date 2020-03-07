package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

func init() {
	rand.Seed(int64(time.Now().Nanosecond()))
}

func Cos(u, v mat.Vector) float64 {
	return 1 - mat.Dot(u, v)/mat.Norm(u, 2)/mat.Norm(v, 2)
}

func PR(A mat.Matrix, e, d float64) *mat.VecDense {
	A_hat := mat.DenseCopyOf(A)
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
		v := mat.NewDense(1, k, A_hat.RawRowView(i))
		n := mat.Sum(v)
		if n == 0 {
			n = 1
		} else {
			n = 1 / n
		}
		v.Scale(n, v)
	}
	n := float64(k)
	A_hat.Apply(func(i, j int, x float64) float64 {
		return x*(1-d) + (d / n)
	}, A_hat)
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

type Tokens []string

func Tokenize(raw string, stops []string) (T Tokens) {
	stops = append(stops, "")
	S := make(map[string]struct{}, len(stops))
	for _, t := range stops {
		S[t] = struct{}{}
	}
	delim := regexp.MustCompile(`[\s\p{Mn}\p{P}0-9]+`)
	T = delim.Split(raw, -1)
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

func (T Tokens) String() string {
	return "#" + strings.Join([]string(T), "-")
}

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
func (T Tokens) NGrams(n int, f func(Tokens)) {
	i := 0
	for i <= len(T)-n {
		f(T[i : i+n])
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
	v := mat.NewVecDense(D.V(), tf)
	v.ScaleVec(1/mat.Sum(v), v)
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
	v := mat.NewVecDense(D.V(), idf)
	v.ScaleVec(1/mat.Sum(v), v)
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
				A.Set(i, k, A.At(i, k)+j)
			}
		}
	})
	A.Apply(func(i, j int, x float64) float64 {
		return math.Log((x + 1) / tf[i] / tf[j])
	}, A)
	return
}

type Traker struct {
	Doc
	TF, IDF, TR []float64
}

func TrakeFrom(D Doc, s, j int) Traker {
	tf := D.TF()
	A := D.PMI(j, tf)
	R := Traker{
		Doc: D,
		TF:  tf,
		IDF: D.IDF(s),
		TR:  PR(A, 1e-6, 0.3).RawVector().Data,
	}
	return R
}

func (R Traker) Score(S Tokens) (x float64) {
	for _, t := range S {
		i := R.Vocab[t]
		x += 0.3 * R.IDF[i] * R.TF[i]
		x += 0.7 * R.TR[i]
	}
	x /= float64(len(S))
	return
}

type Keygrams struct {
	Candidates []Tokens
	Ranks      []float64
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
	F := make(map[string]float64, 1024)
	K.Candidates = make([]Tokens, 0, 1024)
	R.RecNGrams(n, func(S Tokens) {
		k := S.String()
		if _, ok := F[k]; ok {
			F[k] += 1
		} else {
			K.Candidates = append(K.Candidates, S)
			F[k] = 1
		}
	})
	K.Ranks = make([]float64, len(K.Candidates))
	for i, S := range K.Candidates {
		K.Ranks[i] = R.Score(S) / math.Log(F[S.String()]+1) / float64(len(S)+1)
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
	stops := []string{
		"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
		"you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
		"yourselves", "he", "him", "his", "himself", "she", "she's", "her",
		"hers", "herself", "it", "it's", "its", "itself", "they", "them",
		"their", "theirs", "themselves", "what", "which", "who", "whom",
		"this", "that", "that'll", "these", "those", "am", "is", "are", "was",
		"were", "be", "been", "being", "have", "has", "had", "having", "do",
		"does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
		"because", "as", "until", "while", "of", "at", "by", "for", "with",
		"about", "against", "between", "into", "through", "during", "before",
		"after", "above", "below", "to", "from", "up", "down", "in", "out",
		"on", "off", "over", "under", "again", "further", "then", "once",
		"here", "there", "when", "where", "why", "how", "all", "any", "both",
		"each", "few", "more", "most", "other", "some", "such", "no", "nor",
		"not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
		"can", "will", "just", "don", "don't", "should", "should've", "now",
		"d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
		"couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
		"hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma",
		"mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan",
		"shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't",
		"won", "won't", "wouldn", "wouldn't",
	}
	T := Tokenize(strings.ToLower(string(buf)), stops)
	D := DocFrom(T)
	R := TrakeFrom(D, 32, 5)
	K := R.Keygrams(2)
	K.Sort()
	fmt.Println(K.Candidates[:10])
}
