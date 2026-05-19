"""
clustering.py — KMeans Clustering pipeline.
Uses banner() and save_fig() from src.utils (no local copies).
"""
from __future__ import annotations

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

from src.config import (CLUSTER_PLOT_2D, CLUSTER_PLOT_BAR, CLUSTERING_FEATURES,
                         ELBOW_K_MAX, ELBOW_K_MIN, ELBOW_PLOT_PATH, KMEANS_INIT,
                         KMEANS_MAX_ITER, KMEANS_N_CLUSTERS, KMEANS_N_INIT,
                         KMEANS_SEED, MODEL_CLUSTERING_PATH, MODELS_DIR,
                         PLOT_DPI, PLOT_STYLE, PROCESSED_DATASET_PATH)
from src.logger import get_logger
from src.utils import banner, save_fig          # ← single shared source

logger = get_logger(__name__)
plt.style.use(PLOT_STYLE)


def load_processed_data(path=None) -> pd.DataFrame:
    banner("Step 1 — Load Preprocessed Dataset", logger)
    target = path or PROCESSED_DATASET_PATH
    if not target.exists():
        raise FileNotFoundError(f"Processed dataset not found at {target}.")
    df = pd.read_csv(target)
    logger.info("Loaded %d rows × %d columns from %s", *df.shape, target)
    return df


def prepare_features(df, feature_cols=None):
    banner("Step 2 — Prepare & Scale Features", logger)
    cols    = feature_cols or CLUSTERING_FEATURES
    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning("Feature columns not found (skipped): %s", absent)
    if not present:
        raise ValueError("No valid feature columns found for clustering.")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[present].values)
    logger.info("Features: %d → %s", len(present), present)
    return X_scaled, present, scaler


def run_elbow_method(X_scaled, k_min=ELBOW_K_MIN, k_max=ELBOW_K_MAX,
                     save_path=ELBOW_PLOT_PATH) -> int:
    banner("Step 3 — Elbow Method  (KElbowVisualizer)", logger)
    logger.info("Evaluating k = %d … %d", k_min, k_max - 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    base_model = KMeans(init=KMEANS_INIT, n_init=KMEANS_N_INIT,
                        max_iter=KMEANS_MAX_ITER, random_state=KMEANS_SEED)
    viz = KElbowVisualizer(base_model, k=(k_min, k_max), timings=False,
                           locate_elbow=True, ax=ax, force_model=True)
    viz.fit(X_scaled)

    if viz.elbow_value_ is not None:
        optimal_k = int(viz.elbow_value_)
    else:
        scores = list(viz.k_scores_)
        k_vals = list(range(k_min, k_max))
        drops  = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        optimal_k = int(k_vals[int(np.argmax(drops))])
        logger.info("Manual max-drop fallback → k = %d", optimal_k)

    ax.set_title(f"Elbow Method — Distortion Score  (optimal k = {optimal_k})",
                 fontsize=13, fontweight="bold")
    save_fig(fig, save_path, logger=logger)
    logger.info("Optimal k : %d", optimal_k)
    return optimal_k


def train_kmeans(X_scaled, n_clusters=KMEANS_N_CLUSTERS):
    banner(f"Step 4 — Train KMeans  (k = {n_clusters})", logger)
    model = KMeans(n_clusters=n_clusters, init=KMEANS_INIT, n_init=KMEANS_N_INIT,
                   max_iter=KMEANS_MAX_ITER, random_state=KMEANS_SEED)
    model.fit(X_scaled)
    labels = model.labels_
    logger.info("Inertia: %.4f  |  Iterations: %d", model.inertia_, model.n_iter_)
    for k, c in zip(*np.unique(labels, return_counts=True)):
        logger.info("  Cluster %d → %d rows (%.1f%%)", k, c, c/len(labels)*100)
    return model, labels


def evaluate_clusters(df, labels, feature_cols):
    banner("Step 5 — Cluster Evaluation", logger)
    present = [c for c in feature_cols if c in df.columns]
    profile = (df[present].copy().assign(cluster=labels)
                 .groupby("cluster").agg(["mean","count"]))
    profile.columns = ["_".join(c) for c in profile.columns]
    logger.info("\n%s", profile.T.to_string())
    return profile


def plot_clusters(X_scaled, labels, df_original, feature_cols):
    banner("Step 6 — Cluster Visualisations", logger)
    n_clusters = len(np.unique(labels))
    palette    = plt.cm.tab10.colors[:n_clusters]

    pca  = PCA(n_components=2, random_state=KMEANS_SEED)
    X_2d = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(X_2d[mask,0], X_2d[mask,1], s=25, alpha=0.6,
                   color=palette[k], label=f"Cluster {k}")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title("K-Means Clusters — PCA 2-D Projection", fontsize=13, fontweight="bold")
    ax.legend(title="Cluster")
    save_fig(fig, CLUSTER_PLOT_2D, logger=logger)

    present = [c for c in feature_cols if c in df_original.columns]
    profile_norm = (lambda p: (p-p.min())/(p.max()-p.min()+1e-9))(
        df_original[present].assign(cluster=labels).groupby("cluster").mean())
    n_feat, bar_w, x = len(present), 0.8/n_clusters, np.arange(len(present))
    fig, ax = plt.subplots(figsize=(max(12, n_feat*1.2), 5))
    for k in range(n_clusters):
        offset = (k - n_clusters/2 + 0.5) * bar_w
        ax.bar(x+offset, profile_norm.iloc[k], width=bar_w,
               color=palette[k], label=f"Cluster {k}", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
    ax.set_title("Cluster Feature Profiles (normalised mean)", fontsize=13, fontweight="bold")
    ax.legend(title="Cluster")
    save_fig(fig, CLUSTER_PLOT_BAR, logger=logger)


def save_model(model, scaler):
    banner("Step 7 — Save Model  (joblib.dump → model_clustering.pkl)", logger)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artefact = {"model": model, "scaler": scaler,
                "k": model.n_clusters, "features": CLUSTERING_FEATURES}
    joblib.dump(artefact, MODEL_CLUSTERING_PATH)
    logger.info("Model saved → %s  (k=%d, inertia=%.4f)",
                MODEL_CLUSTERING_PATH, model.n_clusters, model.inertia_)


def run_full_clustering(df=None, use_elbow=True) -> dict:
    banner("Clustering Pipeline  —  START", logger)
    if df is None:
        df = load_processed_data()
    X_scaled, feature_cols, scaler = prepare_features(df)
    optimal_k = (run_elbow_method(X_scaled) if use_elbow
                 else (logger.info("Skipping elbow → k=%d", KMEANS_N_CLUSTERS) or KMEANS_N_CLUSTERS))
    model, labels     = train_kmeans(X_scaled, n_clusters=optimal_k)
    cluster_profile   = evaluate_clusters(df, labels, feature_cols)
    plot_clusters(X_scaled, labels, df, feature_cols)
    save_model(model, scaler)
    banner("Clustering Pipeline  —  COMPLETE", logger)
    return {"df": df, "X_scaled": X_scaled, "feature_cols": feature_cols,
            "scaler": scaler, "optimal_k": optimal_k, "model": model,
            "labels": labels, "cluster_profile": cluster_profile}
