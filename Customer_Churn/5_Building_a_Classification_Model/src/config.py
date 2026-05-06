"""
config.py
---------
Central configuration for the Retail EDA project.
All tuneable constants live here — nothing is hardcoded elsewhere.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_FILENAME = "retail_customers.csv"
DATASET_PATH     = DATA_DIR / DATASET_FILENAME

# Number of rows shown by head()
HEAD_ROWS = 10

# ── Synthetic data generation (used when no real CSV is present) ──────────────
SYNTHETIC_N_ROWS    = 1_000
SYNTHETIC_SEED      = 42
SYNTHETIC_DUPE_ROWS = 20        # intentional duplicate rows injected for demo

# ── EDA display ──────────────────────────────────────────────────────────────
DESCRIBE_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# ── Plotting ──────────────────────────────────────────────────────────────────
PLOT_STYLE        = "seaborn-v0_8-whitegrid"
PLOT_DPI          = 150
PLOT_FIGSIZE_WIDE = (14, 5)
PLOT_FIGSIZE_SQ   = (8, 6)

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Columns to drop: IDs, addresses, and date fields (no predictive value)
COLUMNS_TO_DROP = [
    "customer_id",
    "transaction_id",
    "account_id",
    "device_id",
    "ip_address",
    "merchant_id",
    "transaction_date",
]

# Categorical columns that will be label-encoded
CATEGORICAL_COLUMNS = ["gender", "region", "loyalty_tier"]

# Where to persist the cleaned + encoded dataset
PROCESSED_DATASET_PATH = DATA_DIR / "retail_customers_processed.csv"

# ── Clustering ────────────────────────────────────────────────────────────────
# Feature columns used as input to KMeans (excludes target 'churned')
CLUSTERING_FEATURES = [
    "age",
    "tenure_months",
    "annual_spend_usd",
    "num_purchases",
    "avg_order_value",
    "returns_count",
    "last_purchase_days",
    "email_opt_in",
    "gender",
    "region",
    "loyalty_tier",
]

# KElbowVisualizer — range of k to evaluate
ELBOW_K_MIN = 2
ELBOW_K_MAX = 11          # exclusive upper bound (evaluates 2–10)

# Final KMeans configuration
KMEANS_N_CLUSTERS = 4     # updated automatically after elbow analysis if desired
KMEANS_INIT       = "k-means++"
KMEANS_N_INIT     = 10
KMEANS_MAX_ITER   = 300
KMEANS_SEED       = 42

# Where to save the trained model (joblib format, required name for evaluation)
MODELS_DIR            = BASE_DIR / "models"
MODEL_CLUSTERING_PATH = MODELS_DIR / "model_clustering.pkl"

# Cluster visualisation output
CLUSTER_PLOT_2D  = OUTPUT_DIR / "06_cluster_scatter_2d.png"
CLUSTER_PLOT_BAR = OUTPUT_DIR / "07_cluster_profiles.png"
ELBOW_PLOT_PATH  = OUTPUT_DIR / "08_elbow_method.png"

# ── Cluster Interpretation ────────────────────────────────────────────────────
# Aggregation stats shown in the descriptive analysis
INTERPRETATION_AGG_FUNCS = ["mean", "min", "max", "std", "median"]

# Numeric features included in the descriptive analysis table
# (subset of CLUSTERING_FEATURES — only continuous / ordinal columns)
NUMERIC_FEATURES_FOR_ANALYSIS = [
    "age",
    "tenure_months",
    "annual_spend_usd",
    "num_purchases",
    "avg_order_value",
    "returns_count",
    "last_purchase_days",
]

# Column name written to the exported CSV for cluster labels
TARGET_COLUMN_NAME = "Target"

# Output paths for this stage
INTERPRETATION_STATS_CSV  = OUTPUT_DIR / "09_cluster_descriptive_stats.csv"
INTERPRETATION_PROFILE_CSV = OUTPUT_DIR / "10_cluster_profiles_full.csv"
INTERPRETATION_HEATMAP_PNG = OUTPUT_DIR / "11_cluster_heatmap.png"
INTERPRETATION_RADAR_PNG   = OUTPUT_DIR / "12_cluster_radar.png"
LABELED_DATASET_PATH       = DATA_DIR   / "retail_customers_labeled.csv"

# ── Classification ────────────────────────────────────────────────────────────
# Input: labeled dataset produced by cluster_interpreter (has "Target" column)
CLASSIFICATION_INPUT_PATH = LABELED_DATASET_PATH   # data/retail_customers_labeled.csv

# Feature columns (all except the target)
# "churned" is kept as a feature — it was present at clustering time
CLASSIFICATION_LABEL_COL = TARGET_COLUMN_NAME      # "Target"

# Train / test split
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
STRATIFY       = True    # stratify by target to preserve class balance

# Decision Tree hyperparameters
DT_MAX_DEPTH       = None    # None → grow until leaves are pure (then prune via ccp)
DT_MIN_SAMPLES_SPLIT = 2
DT_MIN_SAMPLES_LEAF  = 1
DT_CRITERION         = "gini"
DT_SEED              = 42

# Model output — named exactly as required by the reviewer
DT_MODEL_FILENAME = "decision_tree_model.h5"
DT_MODEL_PATH     = MODELS_DIR / DT_MODEL_FILENAME

# Classification report / visualisation outputs
DT_REPORT_CSV          = OUTPUT_DIR / "13_classification_report.csv"
DT_CONFUSION_PNG       = OUTPUT_DIR / "14_confusion_matrix.png"
DT_FEATURE_IMP_PNG     = OUTPUT_DIR / "15_feature_importance.png"
DT_TREE_PNG            = OUTPUT_DIR / "16_decision_tree_viz.png"
