import warnings
warnings.filterwarnings('ignore')

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def create_sample_corpus():
    """Create a sample text corpus for training Word2Vec"""
    sentences = [
        "the cat sits on the mat",
        "the dog plays in the park",
        "cat and dog are animals",
        "the boy plays with the ball",
        "the girl reads a book",
        "the teacher teaches students in school",
        "students learn from books",
        "python is a programming language",
        "machine learning uses algorithms",
        "deep learning is part of machine learning",
        "neural networks are powerful",
        "the sun shines bright today",
        "the moon comes at night",
        "stars twinkle in the sky",
        "birds fly in the sky",
        "fish swim in the water",
        "trees grow in the forest",
        "flowers bloom in spring",
        "winter is cold and snowy",
        "summer is hot and sunny",
        "cars drive on roads",
        "planes fly in the air",
        "trains run on tracks",
        "computers process information",
        "phones connect people worldwide",
    ]
    return [sent.split() for sent in sentences]


def train_word2vec(sentences, vector_size=50, window=3, min_count=1):
    """Train Word2Vec model"""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )
    return model


def get_word_vectors(model, words=None):
    """Extract word vectors from model"""
    if words is None:
        words = list(model.wv.key_to_index.keys())

    vectors = []
    valid_words = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)

    return np.array(vectors), valid_words


def cluster_words(vectors, n_clusters=5):
    """Cluster word vectors using K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    return labels, kmeans


def reduce_dimensions_pca(vectors, n_components=2):
    """Reduce dimensions using PCA"""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(vectors)
    variance = pca.explained_variance_ratio_
    return reduced, variance


def reduce_dimensions_tsne(vectors, n_components=2):
    """Reduce dimensions using t-SNE"""
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(vectors)-1))
    reduced = tsne.fit_transform(vectors)
    return reduced


def plot_clusters(reduced_vectors, words, labels, method_name, out_path, variance=None):
    """Plot word clusters and save to file"""
    plt.figure(figsize=(14, 10))

    n_clusters = len(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        mask = labels == i
        plt.scatter(
            reduced_vectors[mask, 0],
            reduced_vectors[mask, 1],
            c=[colors[i]],
            label=f'Cluster {i}',
            alpha=0.6,
            s=100
        )

    for i, word in enumerate(words):
        plt.annotate(
            word,
            (reduced_vectors[i, 0], reduced_vectors[i, 1]),
            fontsize=9,
            alpha=0.8
        )

    title = f'Word Clustering using {method_name}'
    if variance is not None:
        title += f'\n(Variance explained: {variance[0]:.2%} + {variance[1]:.2%} = {sum(variance):.2%})'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'{method_name} Component 1', fontsize=11)
    plt.ylabel(f'{method_name} Component 2', fontsize=11)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_clusters(words, labels, n_clusters):
    """Print words in each cluster"""
    print("\nWord Clusters:")
    print("=" * 80)

    for i in range(n_clusters):
        cluster_words = [words[j] for j in range(len(words)) if labels[j] == i]
        print(f"\nCluster {i} ({len(cluster_words)} words):")
        print(f"  {', '.join(sorted(cluster_words))}")


def main():
    print("=" * 80)
    print("WORD CLUSTERING WITH WORD2VEC EMBEDDINGS")
    print("=" * 80)

    print("\n[1/6] Creating sample corpus...")
    sentences = create_sample_corpus()
    print(f"✓ Corpus: {len(sentences)} sentences")

    vocab = set(word for sent in sentences for word in sent)
    print(f"✓ Vocabulary: {len(vocab)} unique words")

    print("\nSample sentences:")
    for i, sent in enumerate(sentences[:5], 1):
        print(f"  {i}. {' '.join(sent)}")
    print(f"  ... and {len(sentences)-5} more")

    print("\n[2/6] Training Word2Vec model...")
    vector_size = 50
    model = train_word2vec(sentences, vector_size=vector_size, window=3, min_count=1)
    print(f"✓ Model trained with {vector_size}-dimensional embeddings")
    print(f"✓ Vocabulary size: {len(model.wv)}")

    print("\n[3/6] Extracting word vectors...")
    vectors, words = get_word_vectors(model)
    print(f"✓ Extracted {len(vectors)} word vectors")
    print(f"✓ Vector shape: {vectors.shape}")

    print("\n[4/6] Clustering words with K-Means...")
    n_clusters = 5
    labels, kmeans = cluster_words(vectors, n_clusters=n_clusters)
    print(f"✓ Created {n_clusters} clusters")
    print(f"✓ Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

    print_clusters(words, labels, n_clusters)

    print("\n[5/6] Reducing dimensions with PCA...")
    pca_vectors, variance = reduce_dimensions_pca(vectors, n_components=2)
    print(f"✓ Reduced to 2D using PCA")
    print(f"✓ Variance explained by PC1: {variance[0]:.2%}")
    print(f"✓ Variance explained by PC2: {variance[1]:.2%}")
    print(f"✓ Total variance explained: {sum(variance):.2%}")

    print("\n[6/6] Reducing dimensions with t-SNE...")
    tsne_vectors = reduce_dimensions_tsne(vectors, n_components=2)
    print(f"✓ Reduced to 2D using t-SNE")

    print("\n[7/7] Saving plots...")
    from pathlib import Path
    out_dir = Path(__file__).parent

    plot_clusters(pca_vectors, words, labels, 'PCA', out_dir / 'clusters_pca.png', variance)
    print(f"✓ Saved: clusters_pca.png")

    plot_clusters(tsne_vectors, words, labels, 't-SNE', out_dir / 'clusters_tsne.png')
    print(f"✓ Saved: clusters_tsne.png")

    print("=" * 80)


if __name__ == '__main__':
    main()
