import numpy as np
from sklearn.manifold import TSNE


def prepare_embeddings(data):
    """Prepara os embeddings e metadados a partir do dicionário de dados."""
    all_embeddings = []
    labels = []
    subject_ids = []
    image_ids = []

    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                all_embeddings.append(embedding)
                labels.append(syndrome_id)
                subject_ids.append(subject_id)
                image_ids.append(image_id)

    return np.array(all_embeddings), labels, subject_ids, image_ids


def compute_tsne(embeddings, n_components=2, perplexity=30, random_state=42):
    """Computa a redução de dimensionalidade com t-SNE."""
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    return tsne.fit_transform(embeddings)
