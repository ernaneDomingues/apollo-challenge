import numpy as np


def cosine_distance(vector_a, vector_b):
    """Calculates the cosine distance between two vectors.

    Args:
        vector_a (array-like): First vector
        vector_b (array-like): Second vector

    Returns:
        float: Cosine distance between the two vectors.
    """
    return 1 - np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def euclidean_distance(vector_a, vector_b):
    """Calculates the euclidean distance between two vectors.

    Args:
        vector_a (array-like): First vector
        vector_b (array-like): Second vector

    Returns:
        float: Euclidean distance between the two vectors.
    """
    return np.linalg.norm(np.array(vector_a) - np.array(vector_b))
