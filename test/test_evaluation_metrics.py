import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from evaluation_metrics import evaluate_models, calculate_auc_ovr, calculate_accuracy


class TestEvaluationFunctions(unittest.TestCase):

    def setUp(self):
        """Configuração de dados fictícios para os testes."""
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])  # Classes reais
        self.y_pred_proba = np.array(
            [  # Probabilidades previstas
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
                [0.8, 0.15, 0.05],
                [0.05, 0.9, 0.05],
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
                [0.15, 0.75, 0.1],
            ]
        )
        self.similarities = {"Cosine Similarity": self.y_pred_proba}
        self.num_labels = 3

    def test_calculate_auc_ovr(self):
        """Teste para a função calculate_auc_ovr."""
        one_hot_true = np.eye(self.num_labels)[self.y_true]
        expected_auc = roc_auc_score(one_hot_true, self.y_pred_proba, multi_class="ovr")
        calculated_auc = calculate_auc_ovr(
            self.y_true, self.y_pred_proba, self.num_labels
        )
        self.assertAlmostEqual(calculated_auc, expected_auc, places=5)

    def test_calculate_accuracy(self):
        """Teste para a função calculate_accuracy."""
        y_pred = np.argmax(self.y_pred_proba, axis=1)
        expected_accuracy = accuracy_score(self.y_true, y_pred)
        calculated_accuracy = calculate_accuracy(self.y_true, self.y_pred_proba)
        self.assertAlmostEqual(calculated_accuracy, expected_accuracy, places=5)

    def test_evaluate_models(self):
        """Teste para a função evaluate_models."""
        results = evaluate_models(self.y_true, self.similarities, self.num_labels)

        # Validar o resultado
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["Metric"], "Cosine Similarity")

        # Validar AUC e Acurácia
        expected_auc = calculate_auc_ovr(
            self.y_true, self.y_pred_proba, self.num_labels
        )
        expected_accuracy = calculate_accuracy(self.y_true, self.y_pred_proba)

        self.assertAlmostEqual(results[0]["AUC"], expected_auc, places=5)
        self.assertAlmostEqual(results[0]["Accuracy"], expected_accuracy, places=5)


if __name__ == "__main__":
    unittest.main()
