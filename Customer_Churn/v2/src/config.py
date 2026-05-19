"""
config.py — Central configuration. All tuneable constants live here.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH           = DATA_DIR / "retail_customers.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "retail_customers_processed.csv"
LABELED_DATASET_PATH   = DATA_DIR / "retail_customers_labeled.csv"

# ── Synthetic data ────────────────────────────────────────────────────────────
SYNTHETIC_N_ROWS    = 1_000
SYNTHETIC_SEED      = 42
SYNTHETIC_DUPE_ROWS = 20

# ── EDA ───────────────────────────────────────────────────────────────────────
HEAD_ROWS            = 10
DESCRIBE_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# ── Plotting ──────────────────────────────────────────────────────────────────
PLOT_STYLE        = "seaborn-v0_8-whitegrid"
PLOT_DPI          = 150
PLOT_FIGSIZE_WIDE = (14, 5)
PLOT_FIGSIZE_SQ   = (8, 6)

# ── Preprocessing ─────────────────────────────────────────────────────────────
COLUMNS_TO_DROP     = ["customer_id","transaction_id","account_id",
                        "device_id","ip_address","merchant_id","transaction_date"]
CATEGORICAL_COLUMNS = ["gender", "region", "loyalty_tier"]

# ── Clustering ────────────────────────────────────────────────────────────────
CLUSTERING_FEATURES = ["age","tenure_months","annual_spend_usd","num_purchases",
                        "avg_order_value","returns_count","last_purchase_days",
                        "email_opt_in","gender","region","loyalty_tier"]
ELBOW_K_MIN       = 2
ELBOW_K_MAX       = 11
KMEANS_N_CLUSTERS = 4
KMEANS_INIT       = "k-means++"
KMEANS_N_INIT     = 10
KMEANS_MAX_ITER   = 300
KMEANS_SEED       = 42
MODEL_CLUSTERING_PATH = MODELS_DIR / "model_clustering.pkl"
CLUSTER_PLOT_2D   = OUTPUT_DIR / "06_cluster_scatter_2d.png"
CLUSTER_PLOT_BAR  = OUTPUT_DIR / "07_cluster_profiles.png"
ELBOW_PLOT_PATH   = OUTPUT_DIR / "08_elbow_method.png"

# ── Cluster Interpretation ────────────────────────────────────────────────────
INTERPRETATION_AGG_FUNCS       = ["mean", "min", "max", "std", "median"]
NUMERIC_FEATURES_FOR_ANALYSIS  = ["age","tenure_months","annual_spend_usd",
                                   "num_purchases","avg_order_value",
                                   "returns_count","last_purchase_days"]
TARGET_COLUMN_NAME             = "Target"
INTERPRETATION_STATS_CSV       = OUTPUT_DIR / "09_cluster_descriptive_stats.csv"
INTERPRETATION_PROFILE_CSV     = OUTPUT_DIR / "10_cluster_profiles_full.csv"
INTERPRETATION_HEATMAP_PNG     = OUTPUT_DIR / "11_cluster_heatmap.png"
INTERPRETATION_RADAR_PNG       = OUTPUT_DIR / "12_cluster_radar.png"

# ── Classification ────────────────────────────────────────────────────────────
CLASSIFICATION_INPUT_PATH  = LABELED_DATASET_PATH
CLASSIFICATION_LABEL_COL   = TARGET_COLUMN_NAME
TEST_SIZE                  = 0.20
RANDOM_STATE               = 42
STRATIFY                   = True
DT_MAX_DEPTH               = None
DT_MIN_SAMPLES_SPLIT       = 2
DT_MIN_SAMPLES_LEAF        = 1
DT_CRITERION               = "gini"
DT_SEED                    = 42
DT_MODEL_FILENAME          = "decision_tree_model.h5"
DT_MODEL_PATH              = MODELS_DIR / DT_MODEL_FILENAME
DT_REPORT_CSV              = OUTPUT_DIR / "13_classification_report.csv"
DT_CONFUSION_PNG           = OUTPUT_DIR / "14_confusion_matrix.png"
DT_FEATURE_IMP_PNG         = OUTPUT_DIR / "15_feature_importance.png"
DT_TREE_PNG                = OUTPUT_DIR / "16_decision_tree_viz.png"
