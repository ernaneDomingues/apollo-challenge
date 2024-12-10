import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from similarity_metrics import (
    calculate_cosine_similarity,
    calculate_euclidean_similarity,
    get_top_k_predictions,
)


class TestSimilarityCalculators(unittest.TestCase):
    def setUp(self):
        """Configuração dos dados fictícios para os testes."""
        self.X_train = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        self.X_test = np.array([[1, 0.5], [0.2, 0.8]])
        self.y_train = np.array([0, 1, 0, 1])
        self.k = 2

    def test_calculate_cosine_similarity(self):
        """Teste para verificar a saída da similaridade por cosseno."""
        predictions = calculate_cosine_similarity(
            self.X_train, self.X_test, self.y_train, self.k
        )

        # Verificar dimensões e soma das probabilidades
        self.assertEqual(predictions.shape, (2, 2))
        np.testing.assert_almost_equal(predictions.sum(axis=1), np.ones(2))

    def test_calculate_euclidean_similarity(self):
        """Teste para verificar a saída da similaridade por distância euclidiana."""
        predictions = calculate_euclidean_similarity(
            self.X_train, self.X_test, self.y_train, self.k
        )

        # Verificar dimensões e soma das probabilidades
        self.assertEqual(predictions.shape, (2, 2))
        np.testing.assert_almost_equal(predictions.sum(axis=1), np.ones(2))

    def test_get_top_k_predictions(self):
        """Teste para verificar a saída do cálculo de probabilidades."""
        similarities = np.array(
            [
                [0.9, 0.7, 0.8, 0.5],  # Similaridades do primeiro teste
                [0.4, 0.6, 0.3, 0.8],  # Similaridades do segundo teste
            ]
        )
        predictions = get_top_k_predictions(similarities, self.y_train, self.k)

        # Resultados esperados (baseados nos 2 vizinhos mais próximos)
        expected_predictions = np.array(
            [
                [1.0, 0.0],  # Classe 0 domina o primeiro conjunto
                [0.0, 1.0],  # Classe 1 domina o segundo conjunto
            ]
        )

        # Verificar se as previsões estão corretas
        np.testing.assert_almost_equal(predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()
