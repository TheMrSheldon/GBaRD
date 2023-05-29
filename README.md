<h1 align="center">
	Effective and Efficient Ranking Using a Dual Encoder Approach
</h1>

<p align="center">
  <a href="#abstract">Abstract</a> &nbsp; | &nbsp;
  <a href="#file-structure">File Structure</a> &nbsp; | &nbsp;
  <a href="#installation">Installation</a> &nbsp; | &nbsp;
  <a href="#usage">Usage</a> &nbsp; | &nbsp;
  <a href="#effectiveness-and-efficiency">Effectiveness and Efficiency</a> &nbsp; | &nbsp;
  <a href="#citation">Citation</a>
</p>

<br/>
<p align="center" id="abstract">
	<b>Abstract</b>
</p>
<center>
	<p align="justify" style="max-width: 20cm;">
		Ever since BERT's inception, the Information Retrieval community has worked on harnessing BERT's potential for
		relevance ranking. To this end, the most prevalent approaches are distilling BERT into smaller transformer
		architectures or designing efficient ranking architectures around (distilled versions of) BERT. With our
		<b>G</b>raph <b>Ba</b>sed <b>R</b>anker for <b>D</b>ocuments (GBaRD), we propose replacing MMSE-ColBERT's
		document encoder by distilling it into a vastly smaller, graph-based architecture. GBaRD creates and
		subsequently refines an initial graph-of-word representation of input documents. When training on TREC DL 2019
		Passage, we find that the smallest variant of our architecture works best. That is, with only a single GAT-layer
		and three GCN-layers, GBaRD outperforms and is competitive with the biggest of our baselines, which have seven
		times the parameter count. Due to its simplicity, GBaRD is three times as fast as state-of-the-art BERT-based
		dual encoders and produces 67% less carbon emissions. Our zero-shot experiments on TREC DL 2019 Document show great
    promise that GBaRD can further be employed for document ranking and that the same graph-based architecture may be
    used to replace the query encoding of MMSE-ColBERT as well for even more efficient ranking.
	</p>
</center>
<br/>


## File Structure
We have split the file structure into two main folders. `Experiments` contains all the code we used to run the
experiments we reported on in the thesis. The model definitions are generally not optimized but written for flexibility
such that different variations can be tested easily. `Optimized` on the other hand contains a vastly smaller definition
of our final architecture (i.e., only one attention head and no jumping knowledge).

## Installation

> GBaRD requires [pytorch](https://pytorch.org/) and
> [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Please install them
> manually before installing GBaRD.

To install either run

```
pip install "GBaRD @ git+https://github.com/TheMrSheldon/GBaRD#v1.0.0"
```
or add

```
GBaRD @ git+https://github.com/TheMrSheldon/GBaRD#v1.0.0
```
to your `requirements.txt`.

## Usage
To run these examples, `cd` into the `examples` folder and install `GBaRD` as described above. You will additionally
need pre-trained weights, which you can get by extracting [gbard-ws3_1x1x2-mmse-8mask.zip](https://github.com/TheMrSheldon/GBaRD/releases/download/v1.0.0/gbard-ws3_1x1x2-mmse-8mask.zip)
in a new `examples/pretrained` folder.

Inference on GBaRD is as easy as loading the pre-trained model and calling it on the input. The call-signature is as follows:
```python
queries: str | list[str], docs: str | list[str]
```
Let `gbard` be an instance of GBaRD, then the following semantic applies:
 - `gbard("query", "document")` returns the relevance of "document" to "query". It is a short-hand notation for `gbard(["query"], ["document"])`.
 - `gbard(["q1", "q2", ...], ["doc1", "doc2", ...])` returns the relevance of "doc1" to "q1", "doc2" to "q2", and so on.
 - `gbard("query", ["doc1", "doc2", ...])` returns the relevance of "doc1" to "query" and of "doc2" to "query" and so on. This returns the same results as `gbard(["query", "query", ...], ["doc1", "doc2", ...])` with the added benefit that the tokenization and representation of "query" is only computed once.

### Basic
The following snippet, for example, computes a relevance grade for the document *"this is a document"* to the query
*"this is a query"*.

```python
from gbard import GBaRD

gbard = GBaRD.from_pretrained("pretrained/gbard-ws3_1x1x2-mmse-8mask")
print(gbard("this is a query", "this is a document"))
# Note that this is a short-hand notation for:
# print(gbard(["this is a query"], ["this is a document"]))
```

### Ranking
```python
from optimized.gbard import GBaRD

gbard = GBaRD.from_pretrained("pretrained/gbard-ws3_1x1x2-mmse-8mask")
query = "What is a cat?"
choices = [
    "Cats are four legged mammals",
    "Dogs are four legged mammals",
    "Dogs, like cats, are four legged mammals",
    "Cats are not like dogs",
    "I like cats",
    "Paris is a city in France",
]

# This line, while correct, will compute the representation for the query multiple times...
# scores = gbard([query]*len(choices), choices)
# ... instead we can write this to only compute the querie's representation once
scores = gbard(query, choices)

print(f"Answers sorted by relevance to '{query}':")
for rank, (choice, score) in enumerate(sorted(zip(choices, scores), key=lambda x: x[1], reverse=True)):
    print(f"\t{rank+1}.: [{score:5.1f}] {choice}")
```

### More Examples
For more examples, have a look at the `examples` folder.

> Note that additional dependencies may be needed to run the experiments.
> Run the following commans (without `$`) to install the remaining dependencies:
> ```sh
> $ conda install pandas
> $ pip install pytrec_eval
> ```

## Effectiveness and Efficiency

Effectiveness scores on
[TRECDL2019 Passage](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html#passage-ranking-dataset) as
compared to other state-of-the-art models.
Model               | MRR@10 | nDCG@10 | MAP@1k
--------------------|--------|---------|--------
ColBERT             |  0.874 |   0.722 |  0.445
TCTColBERT          |  0.951 |   0.670 |  0.386
MMSE-ColBERT        |  0.813 |   0.690 |  0.576
MMSE-ColBERT 8 MASK |  0.882 |   0.762 |  0.632
GBaRD               |  0.831 |   0.708 |  0.587

Effectiveness scores on
[TRECDL2019 Document](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html#document-ranking-dataset) as
compared to other state-of-the-art models.
Model               | MRR@10 | nDCG@10 |   MAP
--------------------|--------|---------|--------
PARADE              |  ----- |   0.679 | 0.287
Birch               |  ----- |   0.640 | 0.328
Simplified TinyBERT |  0.955 |   0.670 | 0.280
GBaRD               |  0.958 |   0.595 | 0.548

Efficiency on [TRECDL2019 Passage](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html#passage-ranking-dataset):
Encoding a document using GBaRD takes one third the time and produces one third the emissions of using ColBERT to do the same.

<p align="center">
    <img src="images/efficiency_dark.svg#gh-dark-mode-only">
    <img src="images/efficiency_dark.svg#gh-light-mode-only">
</p>


## Citation
```
@thesis{hagenEffectiveEfficientRanking2023,
  title = {Effective and {{Efficient Ranking}} Using a {{Dual Encoder Approach}}},
  author = {Hagen, Tim},
  copyright = {GPL-3.0}
}
```