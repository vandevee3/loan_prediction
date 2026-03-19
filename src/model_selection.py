from .utils import deserialize_data, serialize_data
from sklearn.metrics import classification_report

def best_model_selection(models_path: str):
    
    for path in models_path:
        model_load = serialize_data(path)

    pass