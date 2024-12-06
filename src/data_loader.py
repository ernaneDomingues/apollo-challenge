import pickle
import numpy as np


def load_data(file_path):
    """Loads data from a pickle file.

    Args:
        file_path (str): Path of the pickle file to be loaded.

    Returns:
        object: Data loaded from the pickle file.
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def explore_data(data):
    """Explores the data provided and displays basic information.

    Args:
        data (dict): Dictonary containing the data to be explored.
    """
    print(f"Keys (syndrome_ids): {len(data)}")
    print(f"Example key: {list(data.keys())[0]}")
    print(f"Sample structure: {data[list(data.keys())[0]]}")


if __name__ == "__main__":
    data = load_data("mini_gm_public_v0.1.p")
    explore_data(data)
