import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import unittest
import numpy as np
import pickle
import tempfile
from sklearn.preprocessing import LabelEncoder
from data_processing import load_data, prepare_data  # Ajuste o caminho conforme necessário

class TestDataFunctions(unittest.TestCase):

    def setUp(self):
        """
        Configuração do ambiente para os testes.
        Cria um conjunto de dados de exemplo em um arquivo pickle temporário.
        """
        self.sample_data = {
            "syndrome_1": {
                "subject_1": {
                    "image_1": [[1, 2, 3], [4, 5, 6]],
                    "image_2": [[7, 8, 9], [10, 11, 12]],
                },
                "subject_2": {
                    "image_3": [[13, 14, 15], [16, 17, 18]],
                },
            },
            "syndrome_2": {
                "subject_3": {
                    "image_4": [[19, 20, 21], [22, 23, 24]],
                },
            },
        }

        # Cria um arquivo temporário e salva os dados em formato pickle
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(self.temp_file.name, 'wb') as file:
            pickle.dump(self.sample_data, file)

    def tearDown(self):
        """
        Remove o arquivo temporário criado após o teste.
        """
        self.temp_file.close()

    def test_load_data(self):
        """Teste para verificar se os dados são carregados corretamente do arquivo pickle."""
        data = load_data(self.temp_file.name)
        self.assertEqual(data, self.sample_data)

    def test_prepare_data(self):
        """Teste para verificar se os dados são processados corretamente."""
        # Carrega os dados e prepara para o modelo
        data = self.sample_data
        flattened_data, labels, label_encoder = prepare_data(data)

        # Verifica se os dados achatados estão corretos
        expected_flattened_data = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ]
        np.testing.assert_array_equal(flattened_data, np.array(expected_flattened_data))

        # Verifica os labels codificados
        expected_labels = [0, 0, 0, 1]
        np.testing.assert_array_equal(labels, np.array(expected_labels))

        # Verifica os labels originais na codificação inversa
        decoded_labels = label_encoder.inverse_transform(labels)
        expected_decoded_labels = ["syndrome_1", "syndrome_1", "syndrome_1", "syndrome_2"]
        self.assertListEqual(decoded_labels.tolist(), expected_decoded_labels)

if __name__ == "__main__":
    unittest.main()
