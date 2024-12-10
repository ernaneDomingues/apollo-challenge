import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import load_data, prepare_data
from src.similarity_metrics import (
    calculate_cosine_similarity,
    calculate_euclidean_similarity,
)
from src.evaluation_metrics import calculate_auc_ovr, calculate_accuracy
from src.results_saving import create_results_table
from src.data_visualization import (
    plot_class_distribution,
    plot_sample_images,
    plot_roc_curve,
    plot_confusion_matrix,
)
from src.visualization import plot_tsne, plot_cluster_comparison
from src.tsne_processor import prepare_embeddings, compute_tsne


FILE_PATH = "mini_gm_public_v0.1.p"


def main():
    # Carregar os dados
    data = load_data(FILE_PATH)
    X, y, label_encoder = prepare_data(data)

    # Visualizações de pré-processamento
    plot_class_distribution(y, label_encoder)
    plot_sample_images(X, y, label_encoder, n_samples=5)
    
    all_embeddings, labels, _, _ = prepare_embeddings(data)    
    embeddings_2d = compute_tsne(all_embeddings)
    
    plot_tsne(embeddings_2d, labels, title="t-SNE Visualization of Embeddings")

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calcular similaridades
    cosine_predictions_proba = calculate_cosine_similarity(
        X_train, X_test, y_train, k=5
    )

    euclidean_predictions_proba = calculate_euclidean_similarity(
        X_train, X_test, y_train, k=5
    )

    # plot_cluster_comparison(X, cosine_predictions_proba, euclidean_predictions_proba)


    # Avaliar métricas
    num_labels = len(np.unique(y))
    auc_cosine = calculate_auc_ovr(y_test, cosine_predictions_proba, num_labels)
    accuracy_cosine = calculate_accuracy(y_test, cosine_predictions_proba)

    auc_euclidean = calculate_auc_ovr(y_test, euclidean_predictions_proba, num_labels)
    accuracy_euclidean = calculate_accuracy(y_test, euclidean_predictions_proba)

    # Criar tabela de resultados
    results = [
        {"K": 5, "Distance": "Cosine", "Accuracy": accuracy_cosine, "AUC": auc_cosine},
        {
            "K": 5,
            "Distance": "Euclidean",
            "Accuracy": accuracy_euclidean,
            "AUC": auc_euclidean,
        },
    ]
    create_results_table(results, "results.txt")

    # Visualizações de pós-processamento
    plot_roc_curve(y_test, cosine_predictions_proba, num_labels, label_encoder)
    plot_confusion_matrix(
        y_test, np.argmax(cosine_predictions_proba, axis=1), label_encoder
    )


if __name__ == "__main__":
    main()
