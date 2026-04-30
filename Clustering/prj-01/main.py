import os
import pandas as pd
from src.preprocessing import preprocess_data
from src.clustering import find_optimal_k, train_kmeans
from src.utils import save_model

# Load or use existing df
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "dataset.csv")

df = pd.read_csv(data_path)

# Preprocess
df_clean = preprocess_data(df)

# Convert to feature (no target in clustering)
X = df_clean

# Find best K (Elbow Method)
k = find_optimal_k(X)
print("Best K:", k)


# Below code only based/tested

# print("Step 1: Load data")
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(BASE_DIR, "data", "dataset.csv")
# model_path = os.path.join(BASE_DIR, "models", "model_clustering.pkl")

# df = pd.read_csv(data_path)

# print("Step 2: Preprocess")
# df_clean = preprocess_data(df)

# print("Step 3: Find K")
# k = find_optimal_k(df_clean)

# print("Step 4: Train model")
# model = train_kmeans(df_clean, k)

# print("Step 5: Save model")
# # save_model(model, "models/model_clustering.pkl")
# save_model(model, model_path)

# print("DONE ✅")