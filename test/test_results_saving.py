import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import unittest
from unittest.mock import mock_open, patch
from tabulate import tabulate
import numpy as np

from results_saving import create_results_table


class TestCreateResultsTable(unittest.TestCase):
    def setUp(self):
        """Configuração de dados fictícios para os testes."""
        self.results = [
            {
                "K": 5,
                "Distance": "Euclidean",
                "Accuracy": 0.85,
                "AUC": np.float64(0.91),
            },
            {"K": 10, "Distance": "Manhattan", "Accuracy": 0.88, "AUC": 0.92},
            {"K": 15, "Distance": "Cosine", "Accuracy": 0.83, "AUC": np.float64(0.89)},
        ]
        self.filename = "results_table.txt"

    @patch("builtins.open", new_callable=mock_open)
    def test_create_results_table(self, mock_file):
        """Teste para verificar se a tabela é criada corretamente e salva em um arquivo."""
        # Chama a função que será testada
        create_results_table(self.results, self.filename)

        # Gera a tabela esperada manualmente
        expected_table = [
            ["K", "Distance", "Accuracy", "AUC"],
            [5, "Euclidean", 0.85, 0.91],
            [10, "Manhattan", 0.88, 0.92],
            [15, "Cosine", 0.83, 0.89],
        ]
        expected_output = tabulate(expected_table, headers="firstrow", tablefmt="grid")

        # Verifica se o arquivo foi aberto com o nome correto
        mock_file.assert_called_once_with(self.filename, "w")

        # Verifica se o conteúdo gravado no arquivo está correto
        mock_file().write.assert_called_once_with(expected_output)


if __name__ == "__main__":
    unittest.main()
