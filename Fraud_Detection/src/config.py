from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"

DATASET_FILENAME = "bank_transactions_data_edited.csv"
DATASET_PATH     = DATA_DIR / DATASET_FILENAME

HEAD_ROWS = 5

PLOT_DPI         = 150

COLUMNS_TO_DROP = [
    "TransactionID",
    "AccountID",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "TransactionDate",
]

CATEGORICAL_COLUMNS = [
    "TransactionType",
    "Channel",
    "CustomerOccupation",
]

PROCESSED_DATASET_PATH = DATA_DIR / "bank_transactions_data_processed.csv"

CLUSTERING_FEATURES = [
    "TransactionAmount",
    "TransactionType",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]

ELBOW_K_MIN = 2
ELBOW_K_MAX = 11

KMEANS_N_CLUSTERS   = 4
KMEANS_INIT         = "k-means++"
KMEANS_MAX_ITER     = 300
KMEANS_N_INIT       = 10
KMEANS_SEED         = 42

MODELS_DIR             = BASE_DIR / "models"
MODEL_CLUSTERING_PATH  = MODELS_DIR / "model_clustering.pkl"

CLUSTER_PLOT_2D  = OUTPUT_DIR / "cluster_scatter_2d.png"
CLUSTER_PLOT_BAR = OUTPUT_DIR / "cluster_profiles.png"
ELBOW_PLOT_PATH  = OUTPUT_DIR / "elbow_method.png"