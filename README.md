
# trake

`trake` is a new take on an old problem: The problem of [Automatic
summarization][autosumm]. The reasons for its inception are twofold:

1. I found approaches that captured summary keywords of a single lexical unit
to be insufficient for my purposes, because many of what one would consider a
"keyword" are not parsed by the most naive means of subdividing plain text into
lexical units as unigrams. For example, from [the aforementioned wiki
page][autosumm], my methods extracted the following bigrams using a
tag-filtered set of tokens `T` in the call `MoodyEmbeddings.keygrams(T,
breakstops=False)`:

```
#state-art #keyphrase-extraction #determinantal-point #bigram-trigram #et-al
#unigram-bigram #multi-document #maximum-entropy #random-walk
```

...many of which, as you may note, despite being multiple lexical units, only
semantically encapsulate a singular concept or entity.  I attempted an
alternative approach to the one run in the main execution path of the script
derived from TextRank, but the candidate sets I was generating were far too
large for this approach to be computationally viable given my similarity
metrics, which were derived from [this formulation][alacarte], as a result of
having to permute many small phrases in order to generate my desired
candidates.

2. The likes of RAKE and TextRank rely on the small size of their
respective candidate sets in order to yield results that speak to the content
of their documents, but in exchange for their small candidate sets and the
resulting precision, they trade away a bit of what I wanted, which was
_con_cision: Keywords that were longer than one lexical unit, but not by too
much. Twitter leaps to mind, with camel-case hashtags that are multiple words
long, but just barely.

## The solution

I took RAKE's approach of generating candidate sets using phrases broken up by
stopwords, and fused it with TextRank's approach, with a unique scoring metric
for each candidate phrase derived from the normalized TextRank and tf—idf of
their constituent unigrams. This implementation is serial and rather slow, but
still useful as a proof of concept and for reference purposes. Future
refinements may include the development of an English-language stoplist, and
additional heuristics are required for unsupervised use of
`MoodyEmbeddings.stoprank` to generate a stoplist, but POS-filtering has proven
effective enough for my intents and purposes that I surmise a static stoplist
can be empirically determined and hard-coded for better performance with no
losses in precision, and somewhat improved recall (e.g. trigrams of terms
joined by `of`, `and`, or other conjunctions).

For those wondering, `MoodyEmbeddings` is named for the guy who wrote [this
blog post][abstain], because I used his SVD pseudo-vector approach to generate
on-the-fly word embeddings in conjunction with [the á-la-carte
approach][alacarte]. The "TextRank" analysis I'm doing in
`MoodyEmbeddings.stoprank` isn't based on running GloVE or word2vec, but rather
derived from his dimensionality-reduction methodology.

[autosumm]: https://wikipedia.org/wiki/Automatic_summarization
[alacarte]: https://www.offconvex.org/2018/09/18/alacarte/
[abstain]: https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/
