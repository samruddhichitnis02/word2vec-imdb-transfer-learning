# Word2Vec Transfer Learning on IMDB Movie Reviews

## Project Overview

This project investigates how pretrained word embeddings adapt to a specialized
domain through embedding-based transfer learning.

A pretrained Google News Word2Vec model is used as the starting point. The
embeddings are then fine-tuned on IMDB movie reviews to study how the semantic
relationships of movie-related words change after exposure to domain-specific
language.

The project combines:

- Natural Language Processing
- Transfer learning
- Word embedding analysis
- Cosine similarity
- Semantic-neighbor retrieval
- Dimensionality reduction
- Reproducible benchmarking

## Why This Project Matters

General-purpose word embeddings learn broad language patterns from large
corpora. However, the meaning and usage of words can change across domains.

For example, words such as `cast`, `score`, `plot`, `screen`, and `review` have
specific meanings and relationships in movie-review text. Fine-tuning allows
the pretrained embeddings to adapt to those domain-specific patterns while
preserving knowledge learned from the original corpus.

This project demonstrates a practical NLP workflow for adapting pretrained
representations to a specialized dataset.

## Objectives

- Load the pretrained `word2vec-google-news-300` model.
- Establish a pretrained embedding baseline.
- Extract the top-3 nearest neighbors for five movie-related words.
- Tokenize and preprocess IMDB movie reviews.
- Fine-tune the pretrained embeddings on IMDB reviews.
- Re-extract nearest neighbors after fine-tuning.
- Compare semantic relationships before and after adaptation.
- Measure vector movement using cosine similarity.
- Identify the most and least shifted words.
- Visualize embedding movement using t-SNE.
- Record runtime, vocabulary, and dataset statistics.

## Target Words

The following words were selected for analysis:

```text
cast
score
plot
screen
review
```

## Methodology

### 1. Pretrained Embedding Baseline

The pretrained Google News Word2Vec model is loaded using Gensim.

For each target word, the model returns the three nearest words based on
cosine similarity.

This establishes the semantic baseline before domain adaptation.

### 2. IMDB Preprocessing

The IMDB Movie Reviews dataset contains 50,000 labeled movie reviews.

The review text is converted into lowercase tokens using Gensim preprocessing.
These tokenized reviews are used as the training corpus for fine-tuning.

### 3. Embedding-Based Transfer Learning

A new Word2Vec model is created using the IMDB vocabulary.

For words present in both the Google News vocabulary and the IMDB vocabulary,
the pretrained Google News vectors are copied into the new model as initial
weights.

The model is then trained on the IMDB reviews.

```text
Google News pretrained embeddings
                ↓
        IMDB vocabulary
                ↓
Initialize shared words with pretrained vectors
                ↓
       Train on IMDB reviews
                ↓
Movie-review-adapted word embeddings
```

This is transfer learning because the model does not start with completely
random vectors for shared words.

## Model Configuration

- Embedding dimension: 300
- Context window: 5
- Minimum word frequency: 5
- Training method: CBOW
- Fine-tuning epochs: 5
- Random seed: 42
- CPU workers: Based on available CPU cores

## Results

### Semantic Neighbor Analysis

The nearest-neighbor comparison is stored in:

```text
results/neighbor_comparison.csv
```

This file contains:

- Target word
- Neighbor rank
- Neighbor before fine-tuning
- Similarity before fine-tuning
- Neighbor after fine-tuning
- Similarity after fine-tuning

### Vector Shift Analysis

The cosine similarity between each original and fine-tuned vector is stored in:

```text
results/vector_shift_results.csv
```

| Word | Original vs. Fine-Tuned Cosine Similarity | Shift Amount |
|---|---:|---:|
| cast | 0.718312 | 0.281688 |
| score | 0.506448 | 0.493552 |
| plot | 0.686552 | 0.313448 |
| screen | 0.605039 | 0.394961 |
| review | 0.506169 | 0.493831 |

### Main Finding

- Most shifted word: `review`
- Least shifted word: `cast`

The `review` vector experienced the largest change after fine-tuning, suggesting
that its contextual usage was strongly affected by the IMDB movie-review corpus.

The `cast` vector experienced the smallest change among the five selected words,
indicating greater stability between the general-domain and movie-review
representations.

## Visualization

The embedding movement visualization is stored in:

```text
figures/embedding_shifts_tsne.png
```

The visualization uses:

- Blue points for original Google News vectors
- Red points for fine-tuned IMDB vectors
- Arrows to show movement between the two representations

t-SNE is used for qualitative visualization. The cosine similarity table is
used for the quantitative measurement of vector movement because t-SNE can
distort distances during dimensionality reduction.

## Benchmarking

The benchmark summary is stored in:

```text
results/benchmark_summary.csv
```

The benchmark records:

- Dataset loading time
- Tokenization time
- Google News model loading time
- Fine-tuning time
- Number of reviews
- Total number of tokens
- Number of unique words
- Google News vocabulary size
- IMDB vocabulary size
- Shared vocabulary size
- Embedding dimension

This makes the experiment easier to reproduce and evaluate across different
machines.

## Reproducibility

The notebook uses a fixed random seed:

```python
seed = 42
```

The following configuration should remain unchanged when reproducing the
experiment:

- Dataset
- Tokenization method
- Vector dimension
- Context window
- Minimum word frequency
- Number of epochs
- Random seed

Small differences in runtime may occur depending on hardware, operating system,
number of CPU workers, and library versions.

## Project Structure

```text
word2vec-imdb-transfer-learning/
├── notebooks/
│   └── 01_word2vec_imdb.ipynb
├── results/
│   ├── neighbor_comparison.csv
│   ├── vector_shift_results.csv
│   └── benchmark_summary.csv
├── figures/
│   └── embedding_shifts_tsne.png
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/word2vec-imdb-transfer-learning.git
cd word2vec-imdb-transfer-learning
```

### 2. Create the Conda environment

```bash
conda create -n my_env python=3.10
conda activate my_env
```

### 3. Install dependencies

```bash
python -m pip install gensim datasets nltk pandas numpy scikit-learn matplotlib seaborn psutil jupyter ipykernel
```

### 4. Open the notebook

Open:

```text
notebooks/01_word2vec_imdb.ipynb
```

Select the `my_env` Python kernel and run the notebook cells in order.

## Important Model Note

The Google News Word2Vec model is approximately 1.6 GB and is downloaded
automatically the first time the notebook runs.

It is intentionally not included in this repository because of its large size.
The model is stored in the local Gensim cache after downloading.

## Skills Demonstrated

- Python
- Natural Language Processing
- Word2Vec
- Embedding-based transfer learning
- Pretrained model adaptation
- Cosine similarity
- Semantic-neighbor analysis
- IMDB text preprocessing
- t-SNE visualization
- Experimental benchmarking
- Reproducible machine-learning workflows
- Results management with Pandas

## Limitations

- The Google News model is large and requires significant disk space.
- Fine-tuning was performed using a limited number of epochs for practical
  runtime.
- t-SNE provides a qualitative two-dimensional visualization and does not
  preserve all original distances.
- Results may vary slightly across hardware and library versions.
- This project analyzes embedding movement but does not train a sentiment
  classification model.

## Conclusion

The experiment shows that pretrained embeddings can be adapted to a specialized
domain using transfer learning.

Fine-tuning on IMDB reviews changed the semantic representations of the target
words, with `review` showing the largest measured shift and `cast` showing the
smallest. The combination of neighbor analysis, cosine similarity, t-SNE, and
benchmarking provides both qualitative and quantitative evidence of domain
adaptation.