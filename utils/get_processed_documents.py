import os

def get_processed_documents(user_path):
    """Returns a list of already processed document names."""
    vector_stores_path = os.path.join(user_path, "vector_stores")
    if os.path.exists(vector_stores_path):
        return [name for name in os.listdir(vector_stores_path) if os.path.isdir(os.path.join(vector_stores_path, name))]
    return []