import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Carrega os dados a partir de um arquivo pickle."""
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data

def prepare_data(data):
    """
    Prepara os dados para o modelo: 
    - Flatten das imagens 
    - Geração de labels.
    """
    flattened_data = []
    labels = []

    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, image_vector in images.items():
                flattened_data.append(np.array(image_vector).flatten())
                labels.append(syndrome_id)
    
    # Codifica os labels como inteiros
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return np.array(flattened_data), np.array(labels), label_encoder
