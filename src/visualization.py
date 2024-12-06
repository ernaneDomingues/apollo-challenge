import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_tsne(data):
    """Plots a t-SNE visualization of the data provided

    Args:
        data (dict): Dictionary containing the data to be visualized.
    """
    vectors = []

    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, vector in images.items():
                vectors.append(vector)

    vectors = np.array(vectors)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c="blue", label="Syndromes"
    )
    plt.legend(*scatter.legend_elements(), title="Syndromes")
    plt.title("t-SNE of Genetic Syndrome Embeddings")
    plt.show()


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data("mini_gm_public_v0.1.p")
    plot_tsne(data)
