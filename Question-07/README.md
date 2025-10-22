# Word Clustering with Word2Vec

Implements word clustering using Word2Vec embeddings and visualizes clusters with PCA and t-SNE.

## Features

- Trains Word2Vec on sample corpus (25 sentences, 83 words)
- Clusters words using K-Means (k=5)
- Dimensionality reduction with PCA (linear)
- Dimensionality reduction with t-SNE (non-linear)
- Visual cluster plots with labeled words

## Install

```pwsh
pip install gensim scikit-learn matplotlib numpy
```

## Run

```pwsh
python main.py
```

## Output

- Word2Vec training statistics
- K-Means clustering results
- PCA plot showing clusters (with variance explained)
- t-SNE plot showing clusters (better visual separation)
- Cluster membership for all words
- Analysis of results

## Method

1. Train Word2Vec (50-dim embeddings, window=3)
2. Cluster embeddings with K-Means
3. Reduce to 2D with PCA (preserves global structure)
4. Reduce to 2D with t-SNE (preserves local structure)
5. Plot clusters with word labels

Both plots appear automatically showing semantic word groupings.
