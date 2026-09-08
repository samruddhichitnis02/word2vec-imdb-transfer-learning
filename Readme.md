# Word2Vec Transfer Learning on IMDB Movie Reviews

## Overview

This project demonstrates embedding-based transfer learning using a pretrained
Google News Word2Vec model and the IMDB Movie Reviews dataset.

The pretrained embeddings are first evaluated using five movie-related words.
They are then fine-tuned on IMDB reviews and evaluated again to measure how their
meaning changes in the movie-review domain.

## Objectives

- Load the pretrained `word2vec-google-news-300` model.
- Find the top-3 nearest neighbors for:
  - `cast`
  - `score`
  - `plot`
  - `screen`
  - `review`
- Fine-tune the embeddings using IMDB reviews.
- Compare nearest neighbors before and after fine-tuning.
- Measure the cosine similarity between original and fine-tuned vectors.
- Identify the most and least shifted words.
- Visualize embedding movement using t-SNE.
- Record runtime and vocabulary statistics for benchmarking.

## Project Structure

```text
word2vec-imdb-transfer/
├── notebooks/
│   └── 01_word2vec_imdb.ipynb
├── results/
│   ├── neighbor_comparison.csv
│   ├── vector_shift_results.csv
│   └── benchmark_summary.csv
├── figures/
│   └── embedding_shifts_tsne.png
├── src/
├── requirements.txt
├── .gitignore
└── README.md