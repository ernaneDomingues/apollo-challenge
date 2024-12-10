import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve

from data_visualization import (
    plot_class_distribution,
    plot_sample_images,
    plot_roc_curve,
    plot_confusion_matrix,
)

import matplotlib.pyplot as plt


class TestVisualizationFunctions(unittest.TestCase):

    def setUp(self):
        """Configuração do ambiente para os testes."""
        self.labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["Class A", "Class B", "Class C"])

        # Dados fictícios para as funções
        self.X = np.random.rand(8, 32 * 10)  # 8 amostras com dimensões 32x10
        self.y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        self.y_pred_proba = np.random.rand(8, 3)  # Probabilidades para 3 classes
        self.y_pred_proba /= self.y_pred_proba.sum(
            axis=1, keepdims=True
        )  # Normaliza para somar 1
        self.y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1])

    def test_plot_class_distribution(self):
        """Teste para a função plot_class_distribution."""
        try:
            plot_class_distribution(self.labels, self.label_encoder)
        except Exception as e:
            self.fail(f"plot_class_distribution levantou uma exceção: {e}")

    def test_plot_sample_images(self):
        """Teste para a função plot_sample_images."""
        try:
            plot_sample_images(self.X, self.labels, self.label_encoder, n_samples=5)
        except Exception as e:
            self.fail(f"plot_sample_images levantou uma exceção: {e}")

    def test_plot_roc_curve(self):
        """Teste para a função plot_roc_curve."""
        try:
            plot_roc_curve(
                self.y_test,
                self.y_pred_proba,
                num_labels=3,
                label_encoder=self.label_encoder,
            )
        except Exception as e:
            self.fail(f"plot_roc_curve levantou uma exceção: {e}")

    def test_plot_confusion_matrix(self):
        """Teste para a função plot_confusion_matrix."""
        try:
            plot_confusion_matrix(self.y_test, self.y_pred, self.label_encoder)
        except Exception as e:
            self.fail(f"plot_confusion_matrix levantou uma exceção: {e}")

    def tearDown(self):
        """Limpa os plots após cada teste."""
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
