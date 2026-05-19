import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

from src.config import (
    CLUSTERING_FEATURES,
    CLUSTER_PLOT_2D,
    CLUSTER_PLOT_BAR,
    ELBOW_K_MAX,
    ELBOW_K_MIN,
    ELBOW_PLOT_PATH,
    KMEANS_INIT,
    KMEANS_MAX_ITER,
    KMEANS_N_CLUSTERS,
    KMEANS_N_INIT,
    KMEANS_SEED,
    MODEL_CLUSTERING_PATH,
    MODELS_DIR,
    PLOT_DPI,
    PROCESSED_DATASET_PATH,
)
from src.logger import get_logger
from src.utils import banner

logger = get_logger(__name__)

def load_processed_data(path=None) -> pd.DataFrame:
    banner("Step 1 — Load Preprocessed Dataset")

    target = path if path is not None else PROCESSED_DATASET_PATH

    if not target.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {target}.\n"
            "Run main.py first to generate it via the preprocessing pipeline."
        )
    
    df = pd.read_csv(target)
    logger.info("Loaded  : %s", target)
    logger.info("Shape  : %d rows x %d columns", *df.shape)
    logger.info("Columns : %s", list(df.columns))
    return df

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str], StandardScaler]:
    banner("Step 2 — Prepare & Scale Features")

    cols = feature_cols if feature_cols is not None else CLUSTERING_FEATURES
    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]

    if absent:
        logger.warning("Feature columns not found (skipped): %s", absent)
    if not present:
        raise ValueError("No valid feature columns found for clustering.")

    X = df[present].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Features selected  : %d  → %s", len(present), present)
    logger.info("Matrix shape       : %s", X_scaled.shape)
    logger.info("Scaling            : StandardScaler (mean=0) (std=1)")

    return X_scaled, present, scaler

def run_elbow_method(
    X_scaled: np.ndarray,
    k_min: int = ELBOW_K_MIN,
    k_max: int = ELBOW_K_MAX,
    save_path=ELBOW_PLOT_PATH,
) -> int:
    banner("Step 3 — Elbow Method (KElbowVisualizer)")

    logger.info("Evaluating k = %d … %d", k_min, k_max - 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    base_model = KMeans(
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=KMEANS_SEED
    )
    visualizer = KElbowVisualizer(
        base_model,
        k=(k_min, k_max),
        timings=False,
        locate_elbow=True,
        ax=ax,
        force_model=True
    )
    visualizer.fit(X_scaled)

    if visualizer.elbow_value_ is not None:
        optimal_k = int(visualizer.elbow_value_)
        logger.info("Elbow auto-detected at k = %d", optimal_k)
    else:
        scores = list(visualizer.k_scores_)
        k_vals = list(range(k_min, k_max))
        drops  = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        optimal_k = int(k_vals[int(np.argmax(drops))])
        logger.info(
            "Yellowbrick could not auto-detect elbow; "
            "manual max-drop fallback → k = %d", optimal_k
        )

    ax.set_title(
        f"Elbow Method — Distortion Score  (optimal k = {optimal_k})",
        fontsize=13,
        fontweight="bold",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Elbow plot saved  → %s", save_path)
    logger.info("Optimal k         : %d", optimal_k)

    return optimal_k

def train_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = KMEANS_N_CLUSTERS,
) -> tuple[KMeans, np.ndarray]:
    banner(f"Step 4 — Train Kmeans  (k = {n_clusters})")

    model = KMeans(
        n_clusters=n_clusters,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=KMEANS_SEED,
    )
    model.fit(X_scaled)
    labels: np.ndarray = model.labels_

    logger.info("KMeans fitted")
    logger.info("   n_clusters  : %d", model.n_clusters)
    logger.info("   inertia     : %.4f", model.inertia_)
    logger.info("   iterations  : %d", model.n_iter_)

    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Cluster sizes:")
    for k, c in zip(unique, counts):
        logger.info("   Cluster %d → %d rows  (%.1f%%)", k, c, c / len(labels) * 100)
    
    return model, labels

def evaluate_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    banner("Step 5 — Cluster Evaluation  (mean profile per cluster)")

    present = [c for c in feature_cols if c in df.columns]
    profile = (
        df[present]
        .copy()
        .assign(cluster=labels)
        .groupby("cluster")
        .agg(["mean", "count"])
    )

    profile.columns = ["_".join(c).strip() for c in profile.columns]

    logger.info("\n%s", profile.T.to_string())
    return profile


def plot_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    df_original: pd.DataFrame,
    feature_cols: list[str]
) -> None:
    banner("Step 6 — Cluster Visualisations")

    n_clusters  = len(np.unique(labels))
    palette     = plt.cm.tab10.colors[:n_clusters]

    pca     = PCA(n_components=2, random_state=KMEANS_SEED)
    X_2d    = pca.fit_transform(X_scaled)
    var     = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=25, alpha=0.6, color=palette[k], label=f"Cluster {k}",
        )
    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var[0]:.1%} variance)", fontsize=11)
    ax.set_title("K-Means Clusters — PCA 2-D Projection", fontsize=13, fontweight="bold")
    ax.legend(title="Cluster", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(CLUSTER_PLOT_2D, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scatter plot saved → %s", CLUSTER_PLOT_2D)

    present = [c for c in feature_cols if c in df_original.columns]
    profile = (
        df_original[present]
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )

    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    n_features = len(present)
    bar_width  = 0.8 / n_clusters
    x          = np.arange(n_features)

    fig, ax = plt.subplots(figsize=(max(12, n_features * 1.2), 5))
    for k in range(n_clusters):
        offset = (k - n_clusters / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            profile_norm.iloc[k],
            width=bar_width,
            color=palette[k],
            label=f"Cluster {k}",
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Normalised Mean (0-1)")
    ax.set_title(
        "Cluster Feature Profiles (normalised mean per cluster)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(CLUSTER_PLOT_BAR, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Profile plot saved → %s", CLUSTER_PLOT_BAR)

def save_model(model: KMeans, scaler: StandardScaler) -> None:
    banner("Step 7 — Save Model  (joblib.dump → model_clustering.pkl)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artefact = {
        "model":    model,
        "scaler":   scaler,
        "k":        model.n_clusters,
        "features": CLUSTERING_FEATURES,
    }

    joblib.dump(artefact, MODEL_CLUSTERING_PATH)
    logger.info("Model saved → %s", MODEL_CLUSTERING_PATH)
    logger.info("  KMeans k  : %d", model.n_clusters)
    logger.info("  Inertia   : %.4f", model.inertia_)

def run_full_clustering(
    df: pd.DataFrame | None = None,
    use_elbow: bool = True
) -> dict:
    banner("Clustering Pipeline  —  START")

    if df is None:
        df = load_processed_data()

    X_scaled, feature_cols, scaler = prepare_features(df)

    if use_elbow:
        optimal_k = run_elbow_method(X_scaled)
    else:
        optimal_k = KMEANS_N_CLUSTERS
        logger.info("Skipping elbow (use_elbow=False) — using k=%d from config.", optimal_k)

    model, labels = train_kmeans(X_scaled, n_clusters=optimal_k)

    cluster_profile = evaluate_clusters(df, labels, feature_cols)

    plot_clusters(X_scaled, labels, df, feature_cols)

    save_model(model, scaler)

    banner("Clustering Pipeline  —  COMPLETE")
    logger.info("Model artefact : %s", MODEL_CLUSTERING_PATH)
    logger.info("Cluster count  : %d", optimal_k)

    return {
        "df":               df,
        "X_scaled":         X_scaled,
        "feature_cols":     feature_cols,
        "scaler":           scaler,
        "optimal_k":        optimal_k,
        "model":            model,
        "labels":           labels,
        "cluster_profile":  cluster_profile,
    }
