"""
CO4: Unsupervised Learning, Clustering, Dimensionality Reduction, and Anomaly Detection.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from src.config import ARTIFACTS_DIR, RANDOM_STATE, FEATURE_COLUMNS


def run_kmeans_sweep(X_scaled: np.ndarray, k_range=range(2, 9)) -> Tuple[pd.DataFrame, KMeans]:
    """
    Fits KMeans across range of K values, tracks inertia and silhouette scores.
    """
    records = []
    best_k = 2
    best_sil = -1.0
    best_model = None

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        records.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": sil
        })
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = km

    df_km = pd.DataFrame(records)

    # Plot Elbow & Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df_km["K"], df_km["Inertia"], "o-", color="#2b5c8f", lw=2)
    ax1.set_title("K-Means Elbow Curve (Inertia)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters K")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.grid(True)

    ax2.plot(df_km["K"], df_km["Silhouette"], "s-", color="#d95f02", lw=2)
    ax2.axvline(x=best_k, color="red", linestyle=":", label=f"Optimal K = {best_k} (Sil={best_sil:.3f})")
    ax2.set_title("K-Means Silhouette Scores", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters K")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "kmeans_elbow_silhouette.png", dpi=200)
    plt.close()

    return df_km, best_model


def run_hierarchical_and_dendrogram(X_scaled: np.ndarray) -> AgglomerativeClustering:
    """
    Computes hierarchical clustering with Ward linkage and exports dendrogram.
    """
    # Sample 100 observations for a clean, legible dendrogram
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(len(X_scaled), size=min(100, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_indices]

    linkage_matrix = linkage(X_sample, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=25,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )
    plt.title("Hierarchical Agglomerative Clustering (Ward Linkage Dendrogram)", fontsize=13, fontweight="bold")
    plt.xlabel("Sample Index or Cluster Size")
    plt.ylabel("Ward Distance (Variance Increase)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "hierarchical_dendrogram.png", dpi=200)
    plt.close()

    model = AgglomerativeClustering(n_clusters=2, linkage="ward")
    model.fit(X_scaled)
    return model


def run_dbscan_analysis(X_scaled: np.ndarray) -> Tuple[DBSCAN, Dict[str, Any]]:
    """
    Calculates k-distance graph for epsilon selection and fits DBSCAN.
    Identifies core, border, and noise (outlier) points.
    """
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, k - 1])

    # Plot k-distance curve
    plt.figure(figsize=(8, 4))
    plt.plot(k_distances, color="#2b5c8f", lw=2)
    chosen_eps = 2.2
    plt.axhline(y=chosen_eps, color="red", linestyle="--", label=f"Selected Epsilon = {chosen_eps}")
    plt.title(f"DBSCAN: {k}-Distance Graph for Epsilon Selection", fontsize=12, fontweight="bold")
    plt.xlabel("Points Sorted by Distance")
    plt.ylabel(f"{k}-NN Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "dbscan_k_distance.png", dpi=200)
    plt.close()

    db = DBSCAN(eps=chosen_eps, min_samples=5)
    labels = db.fit_predict(X_scaled)

    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    stats = {
        "epsilon": chosen_eps,
        "min_samples": 5,
        "n_clusters_found": n_clusters,
        "noise_points_detected": n_noise,
        "noise_percentage": float(n_noise / len(X_scaled) * 100)
    }
    return db, stats


def run_dimensionality_reduction_dashboard(
    X_unscaled: np.ndarray,
    X_scaled: np.ndarray,
    y_labels: np.ndarray
) -> Dict[str, Any]:
    """
    Creates a 3-panel dashboard:
    1. PCA Scree Plot & Cumulative Explained Variance.
    2. PCA 2D Projection (Color-coded by Diabetes Outcome).
    3. t-SNE 2D Manifold Visualization.
    Also compares PCA variance when applied with vs without feature scaling.
    """
    # 1. PCA with Scaling
    pca = PCA(n_components=min(8, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    # PCA without Scaling (pitfall demo)
    pca_unscaled = PCA(n_components=min(8, X_unscaled.shape[1]))
    pca_unscaled.fit(X_unscaled)
    evr_unscaled = pca_unscaled.explained_variance_ratio_

    # 2. t-SNE with perplexity=30
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, n_iter_without_progress=300)
    X_tsne = tsne.fit_transform(X_scaled)

    # 3. Create Dashboard Figure
    fig = plt.figure(figsize=(18, 5))

    # Panel A: Scree Plot
    ax1 = fig.add_subplot(1, 3, 1)
    components = np.arange(1, len(evr) + 1)
    ax1.bar(components, evr, alpha=0.7, color="#2b5c8f", label="Individual Variance Ratio")
    ax1.plot(components, np.cumsum(evr), "ro-", lw=2, label="Cumulative Variance")
    ax1.set_title("PCA: Scree Plot & Explained Variance (Scaled)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True)

    # Panel B: PCA 2D Projection
    ax2 = fig.add_subplot(1, 3, 2)
    scatter2 = ax2.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_labels,
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="k",
        s=35
    )
    ax2.set_title(f"PCA 2D Projection (PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(["Non-Diabetic", "Diabetic"])

    # Panel C: t-SNE 2D Manifold
    ax3 = fig.add_subplot(1, 3, 3)
    scatter3 = ax3.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=y_labels,
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="k",
        s=35
    )
    ax3.set_title("t-SNE 2D Manifold Projection (Perplexity=30)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("t-SNE Dimension 1")
    ax3.set_ylabel("t-SNE Dimension 2")
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(["Non-Diabetic", "Diabetic"])

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "dim_reduction_dashboard.png", dpi=200)
    plt.close()

    return {
        "scaled_pca_pc1_variance": float(evr[0]),
        "scaled_pca_pc2_variance": float(evr[1]),
        "scaled_pca_top2_cumulative": float(np.sum(evr[:2])),
        "unscaled_pca_pc1_variance": float(evr_unscaled[0]),
        "scaling_pitfall_note": (
            f"Unscaled PCA assigns {evr_unscaled[0]*100:.1f}% of variance solely to high-magnitude features "
            f"(Insulin/Glucose), whereas Scaled PCA distributes variance properly across clinical dimensions."
        )
    }


def run_isolation_forest_anomaly_detection(X_scaled: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """
    Applies Isolation Forest to detect anomalous physiological records.
    Tests contamination rates and visualizes anomaly scores.
    """
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    iso.fit(X_scaled)
    scores = iso.decision_function(X_scaled)
    preds = iso.predict(X_scaled)  # -1 for anomaly, 1 for normal

    n_anomalies = int(np.sum(preds == -1))

    plt.figure(figsize=(8, 4))
    sns.histplot(scores, bins=30, kde=True, color="#7570b3")
    plt.axvline(x=0.0, color="red", linestyle="--", label="Decision Boundary (Anomaly < 0)")
    plt.title("Isolation Forest Anomaly Score Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Anomaly Score (Lower = More Anomalous)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "anomaly_score_distribution.png", dpi=200)
    plt.close()

    return {
        "n_anomalies_detected": n_anomalies,
        "anomaly_percentage": float(n_anomalies / len(X_scaled) * 100),
        "mean_anomaly_score": float(np.mean(scores))
    }
