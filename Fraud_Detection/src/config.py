from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_FILENAME = "bank_transactions_data_edited.csv"
# DATASET_PATH     = DATA_DIR / DATASET_FILENAME
DATASET_PATH     = RAW_DATA_DIR / DATASET_FILENAME

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

PROCESSED_DATASET_FILENAME = "bank_transactions_data_processed.csv"
PROCESSED_DATASET_PATH     = PROCESSED_DATA_DIR / PROCESSED_DATASET_FILENAME

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

MODELS_DIR               = BASE_DIR / "models"
MODEL_CLASSIFICATION_DIR = MODELS_DIR / "classification" 
MODEL_CLUSTERING_DIR     = MODELS_DIR / "clustering"

MODEL_CLUSTER_FILENAME   = "model_clustering.pkl"
MODEL_CLUSTERING_PATH    = MODEL_CLUSTERING_DIR / MODEL_CLUSTER_FILENAME

CLUSTER_PLOT_2D  = OUTPUT_DIR / "cluster_scatter_2d.png"
CLUSTER_PLOT_BAR = OUTPUT_DIR / "cluster_profiles.png"
ELBOW_PLOT_PATH  = OUTPUT_DIR / "elbow_method.png"

INTERPRETATION_AGG_FUNCS = ["mean", "min", "max", "std", "median"]

NUMERIC_FEATURES_FOR_ANALYSIS = [
    "TransactionAmount",
    "TransactionType",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
]

TARGET_COLUMN_NAME = "Target"

INTERPRETATION_STATS_CSV   = OUTPUT_DIR / "09_cluster_descriptive_stats.csv"
INTERPRETATION_PROFILE_CSV = OUTPUT_DIR / "10_cluster_profiles_full.csv"
INTERPRETATION_HEATMAP_PNG = OUTPUT_DIR / "11_cluster_heatmap.png"
INTERPRETATION_RADAR_PNG   = OUTPUT_DIR / "12_cluster_radar.png"

PROCESSED_LABELED_DATASET_FILENAME = "retail_customers_labeled.csv"
LABELED_DATASET_PATH               = PROCESSED_DATA_DIR / PROCESSED_LABELED_DATASET_FILENAME