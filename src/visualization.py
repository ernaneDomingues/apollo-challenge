import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(embeddings_2d, labels, title="t-SNE Visualization"):
    """Cria um gráfico 2D para os embeddings reduzidos com t-SNE."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.7,
    )
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Labels", loc="best", fontsize=10)
    plt.grid(True)
    plt.show()


def plot_cluster_comparison(X, cosine_predictions_proba, euclidean_predictions_proba):
    """
    Plota gráficos comparativos de clusters baseados nas probabilidades de dois modelos.

    Args:
        X: Dados originais (antes da redução de dimensionalidade).
        cosine_predictions_proba: Probabilidades do modelo baseado em cosseno.
        euclidean_predictions_proba: Probabilidades do modelo baseado em euclideana.
    """
    # Ajuste dinâmico do perplexity
    perplexity = min(
        30, X.shape[0] - 1
    )  # Garantir que perplexity é menor que o número de amostras

    # Reduzir dimensionalidade para 2D usando t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Configurar o layout dos gráficos
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Gráfico para Cosine Similarity
    scatter_cosine = axes[0].scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=np.argmax(cosine_predictions_proba, axis=1),  # Cluster de maior probabilidade
        cmap="viridis",
        alpha=0.7,
    )
    axes[0].set_title("Clusters - Cosine Similarity", fontsize=14)
    axes[0].set_xlabel("t-SNE Component 1")
    axes[0].set_ylabel("t-SNE Component 2")
    fig.colorbar(scatter_cosine, ax=axes[0], label="Cluster")

    # Gráfico para Euclidean Distance
    scatter_euclidean = axes[1].scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=np.argmax(
            euclidean_predictions_proba, axis=1
        ),  # Cluster de maior probabilidade
        cmap="plasma",
        alpha=0.7,
    )
    axes[1].set_title("Clusters - Euclidean Distance", fontsize=14)
    axes[1].set_xlabel("t-SNE Component 1")
    fig.colorbar(scatter_euclidean, ax=axes[1], label="Cluster")

    # Mostrar gráficos
    plt.tight_layout()
    plt.show()
