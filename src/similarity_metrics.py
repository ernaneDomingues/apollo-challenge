import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calculate_cosine_similarity(X_train, X_test, y_train, k=5):
    """Calcula similaridade por cosseno e retorna probabilidades."""
    similarities = cosine_similarity(X_test, X_train)
    return get_top_k_predictions(similarities, y_train, k)

def calculate_euclidean_similarity(X_train, X_test, y_train, k=5):
    """Calcula similaridade por distância euclidiana e retorna probabilidades."""
    distances = euclidean_distances(X_test, X_train)
    similarities = 1 / (1 + distances)  # Converte distâncias em similaridades
    return get_top_k_predictions(similarities, y_train, k)

def get_top_k_predictions(similarities, y_train, k):
    """
    Obtém os K vizinhos mais similares e gera probabilidades para cada classe.
    """
    num_classes = len(np.unique(y_train))
    predictions = []

    for sim in similarities:
        top_k_indices = np.argsort(sim)[-k:][::-1]  # Indices dos K maiores
        top_k_labels = y_train[top_k_indices]
        
        probabilities = np.zeros(num_classes)
        for label in top_k_labels:
            probabilities[label] += 1
        
        predictions.append(probabilities / k)  # Normaliza para obter probabilidades

    return np.array(predictions)
