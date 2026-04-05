# utils.py
import os
import pickle

def load_fingerprint_database(path="clustered_data.pkl"):
    print(f"Loading DB from {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None