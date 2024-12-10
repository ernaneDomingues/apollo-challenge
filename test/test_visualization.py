import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import unittest
import numpy as np
from unittest.mock import patch
from visualization import plot_tsne, plot_cluster_comparison


class TestVisualizationFunctions(unittest.TestCase):
    def setUp(self):
        """Configuração de dados fictícios para os testes."""
        self.embeddings_2d = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        self.labels = [0, 1, 0, 1]
        self.X = np.random.rand(10, 5)  # Dados fictícios de alta dimensionalidade
        self.cosine_predictions_proba = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.8, 0.2],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        )
        self.euclidean_predictions_proba = np.array(
            [
                [0.8, 0.2],
                [0.6, 0.4],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.5, 0.5],
                [0.3, 0.7],
            ]
        )

    @patch("matplotlib.pyplot.show")
    def test_plot_tsne(self, mock_show):
        """Teste para verificar se a função plot_tsne executa sem erros."""
        try:
            plot_tsne(self.embeddings_2d, self.labels, title="Teste t-SNE")
        except Exception as e:
            self.fail(f"A função plot_tsne levantou uma exceção inesperada: {e}")

        # Verifica se o gráfico foi chamado
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_cluster_comparison(self, mock_show):
        """Teste para verificar se a função plot_cluster_comparison executa sem erros."""
        try:
            plot_cluster_comparison(
                self.X, self.cosine_predictions_proba, self.euclidean_predictions_proba
            )
        except Exception as e:
            self.fail(
                f"A função plot_cluster_comparison levantou uma exceção inesperada: {e}"
            )

        # Verifica se o gráfico foi chamado
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
