from sklearn.neighbors import KNeighborsClassifier


def knn_classification(train_vectors, train_labels, test_vectors, distance_metric):
    """Classifies the test vectors using the KNN algorithm.

    Args:
        train_vectors (array): Training vectors.
        train_labels (array): Labels of the training vectors.
        test_vectors (array): Test vectors to be classified.
        distance_metric (str): Distance metric to be used by the KNN.

    Returns:
        array: Predict labels for the test vectors.
    """
    knn = KNeighborsClassifier(metric=distance_metric)
    knn.fit(train_vectors, train_labels)
    return knn.predict(test_vectors)
