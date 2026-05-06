import joblib

# def save_model(model, path="models/model_clustering.pkl"):
def save_model(model, path):
    joblib.dump(model, path)