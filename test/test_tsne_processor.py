import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import unittest
import numpy as np
from sklearn.manifold import TSNE
from tsne_processor import prepare_embeddings, compute_tsne


class TestEmbeddingProcessor(unittest.TestCase):
    def setUp(self):
        """Configuração dos dados fictícios para os testes."""
        self.data = {
            0: {  # syndrome_id
                "subject1": {
                    "image1": [0.1, 0.2, 0.3],
                    "image2": [0.4, 0.5, 0.6],
                },
                "subject2": {
                    "image3": [0.7, 0.8, 0.9],
                },
            },
            1: {  # syndrome_id
                "subject3": {
                    "image4": [0.2, 0.3, 0.4],
                    "image5": [0.5, 0.6, 0.7],
                }
            },
        }
        self.embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
            ]
        )

    def test_prepare_embeddings(self):
        """Teste para verificar se os embeddings e metadados são extraídos corretamente."""
        all_embeddings, labels, subject_ids, image_ids = prepare_embeddings(self.data)

        # Verificar embeddings
        np.testing.assert_array_equal(
            all_embeddings,
            self.embeddings,
            "Os embeddings extraídos não correspondem ao esperado.",
        )

        # Verificar labels
        expected_labels = [0, 0, 0, 1, 1]
        self.assertListEqual(
            labels,
            expected_labels,
            "Os labels extraídos não correspondem ao esperado.",
        )

        # Verificar subject_ids
        expected_subject_ids = [
            "subject1",
            "subject1",
            "subject2",
            "subject3",
            "subject3",
        ]
        self.assertListEqual(
            subject_ids,
            expected_subject_ids,
            "Os subject_ids extraídos não correspondem ao esperado.",
        )

        # Verificar image_ids
        expected_image_ids = ["image1", "image2", "image3", "image4", "image5"]
        self.assertListEqual(
            image_ids,
            expected_image_ids,
            "Os image_ids extraídos não correspondem ao esperado.",
        )

    def test_compute_tsne(self):
        """Teste para verificar se o t-SNE é calculado corretamente."""
        tsne_result = compute_tsne(
            self.embeddings, n_components=2, perplexity=2, random_state=42
        )

        # Verificar dimensões do resultado
        self.assertEqual(
            tsne_result.shape,
            (self.embeddings.shape[0], 2),
            "O resultado do t-SNE não tem as dimensões esperadas.",
        )

        # Teste básico: os resultados são diferentes (indicativo de redução de dimensionalidade)
        self.assertFalse(
            np.allclose(self.embeddings[:, :2], tsne_result),
            "O t-SNE não parece ter transformado os embeddings corretamente.",
        )


if __name__ == "__main__":
    unittest.main()
