import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from collections import Counter

# Optional: for visualization
try:
    import umap
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# -----------------------
# Configuration
# -----------------------

EMB_DIR = Path("./encoding_data")

# If you know the exact 7 model names, list them explicitly:
# MODEL_NAMES = ["gpt4", "claude3", "llama3_8b", "llama3_70b", "mistral", "gemini", "other_model"]
MODEL_NAMES: List[str] = []  # leave empty to auto-discover

PCA_DIM = 50          # set to None to skip PCA
N_CLUSTERS = 7
RANDOM_STATE = 42

FIG_DIR = Path("figures")    # directory to save figures
FIG_DPI = 300                # resolution of saved figures
SHOW_FIGS = False            # set True if you also want interactive display


# -----------------------
# Load and combine embeddings
# -----------------------

def load_embeddings_for_model(model_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
    emb_path = EMB_DIR / f"{model_name}_embeddings.npz"
    meta_path = EMB_DIR / f"{model_name}_metadata.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    emb_npz = np.load(emb_path)
    embeddings = emb_npz["embeddings"]
    meta = pd.read_csv(meta_path)

    if len(meta) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch for {model_name}: "
            f"{len(meta)} metadata rows vs {embeddings.shape[0]} embeddings"
        )

    return embeddings, meta


def load_all_models() -> Tuple[np.ndarray, pd.DataFrame]:
    if MODEL_NAMES:
        model_names = MODEL_NAMES
    else:
        model_names = sorted([
            p.name.replace("_embeddings.npz", "")
            for p in EMB_DIR.glob("*_embeddings.npz")
        ])

    if len(model_names) == 0:
        raise RuntimeError(f"No *_embeddings.npz files found in {EMB_DIR}")
    if len(model_names) < N_CLUSTERS:
        print(f"Warning: found {len(model_names)} models, but N_CLUSTERS={N_CLUSTERS}")

    print("Models detected / used:")
    for m in model_names:
        print("  -", m)

    all_embeddings = []
    all_meta = []

    for model_name in model_names:
        emb, meta = load_embeddings_for_model(model_name)
        all_embeddings.append(emb)
        all_meta.append(meta)

    X = np.vstack(all_embeddings)
    meta_all = pd.concat(all_meta, ignore_index=True)

    print(f"\nCombined embeddings shape: {X.shape}")
    print(f"Combined metadata rows: {len(meta_all)}")
    return X, meta_all


# -----------------------
# PCA
# -----------------------

def run_pca(X: np.ndarray, pca_dim: int) -> Tuple[np.ndarray, PCA]:
    print(f"\nRunning PCA to {pca_dim} dimensions...")
    pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
    print(f"Total explained variance (first {pca_dim} components): {pca.explained_variance_ratio_.sum():.4f}")
    return X_pca, pca


# -----------------------
# Clustering and evaluation
# -----------------------

def cluster_and_evaluate(
    X: np.ndarray,
    labels_true: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
):
    print(f"\nClustering with KMeans, n_clusters={n_clusters}...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X)

    # Convert true labels to integers if they are strings
    if labels_true.dtype == object:
        unique_models = np.unique(labels_true)
        model_to_int = {m: i for i, m in enumerate(unique_models)}
        y_true_int = np.array([model_to_int[m] for m in labels_true])
    else:
        y_true_int = labels_true

    ari = adjusted_rand_score(y_true_int, cluster_labels)
    nmi = normalized_mutual_info_score(y_true_int, cluster_labels)
    sil = silhouette_score(X, cluster_labels)

    print("\nClustering evaluation:")
    print(f"  Adjusted Rand Index (ARI):            {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI):  {nmi:.4f}")
    print(f"  Silhouette score:                     {sil:.4f}")

    print("\nCluster composition (per cluster):")
    df_eval = pd.DataFrame({
        "cluster": cluster_labels,
        "true_model": labels_true
    })

    cluster_summaries = []
    for c in range(n_clusters):
        df_c = df_eval[df_eval["cluster"] == c]
        total_c = len(df_c)
        if total_c == 0:
            print(f"  Cluster {c}: empty")
            continue

        counts = Counter(df_c["true_model"])
        most_common_model, most_common_count = counts.most_common(1)[0]
        purity = most_common_count / total_c

        print(f"  Cluster {c}: size={total_c}, purity={purity:.3f}, "
              f"top model={most_common_model} ({most_common_count})")

        cluster_summaries.append({
            "cluster": c,
            "size": total_c,
            "purity": purity,
            "top_model": most_common_model
        })

    cluster_summary_df = pd.DataFrame(cluster_summaries)
    return cluster_labels, cluster_summary_df


# -----------------------
# Visualization helpers
# -----------------------

def ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename_base: str):
    """
    Save a matplotlib figure in PNG (and optionally PDF) with consistent naming.
    """
    ensure_fig_dir()
    png_path = FIG_DIR / f"{filename_base}.png"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved figure to: {png_path}")

    # Uncomment if you also want PDF
    # pdf_path = FIG_DIR / f"{filename_base}.pdf"
    # fig.savefig(pdf_path, dpi=FIG_DPI, bbox_inches="tight")
    # print(f"  Saved figure to: {pdf_path}")


def visualize_2d(
    X: np.ndarray,
    labels_true: np.ndarray,
    cluster_labels: np.ndarray,
    title_suffix: str = ""
):
    """
    Reduce to 2D with UMAP and save:
      - one figure colored by true model labels
      - one figure colored by cluster labels
    """
    if not HAS_UMAP:
        print("UMAP/matplotlib not available, skipping visualization.")
        return

    print("\nRunning UMAP 2D projection for visualization...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=RANDOM_STATE,
        n_neighbors=15,
        min_dist=0.1,
    )
    X_2d = reducer.fit_transform(X)

    unique_models = np.unique(labels_true)
    model_to_color = {m: i for i, m in enumerate(unique_models)}
    colors_true = np.array([model_to_color[m] for m in labels_true])

    # ----- Figure 1: colored by true model -----
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    scatter1 = ax1.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=colors_true, cmap="tab10", s=8, alpha=0.7
    )
    ax1.set_title(f"UMAP: True Black-Box Model Labels {title_suffix}", fontsize=14)
    ax1.set_xlabel("UMAP dimension 1", fontsize=12)
    ax1.set_ylabel("UMAP dimension 2", fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    handles1, _ = scatter1.legend_elements()
    ax1.legend(handles1, unique_models, title="Model", fontsize="small", loc="best")

    filename_base_true = f"umap_true_models_{title_suffix.replace(' ', '_').lower()}"
    save_figure(fig1, filename_base_true)
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig1)

    # ----- Figure 2: colored by cluster -----
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    scatter2 = ax2.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=cluster_labels, cmap="tab10", s=8, alpha=0.7
    )
    ax2.set_title(f"UMAP: KMeans Clusters {title_suffix}", fontsize=14)
    ax2.set_xlabel("UMAP dimension 1", fontsize=12)
    ax2.set_ylabel("UMAP dimension 2", fontsize=12)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

    handles2, _ = scatter2.legend_elements()
    ax2.legend(handles2, [f"Cluster {c}" for c in range(len(handles2))],
               title="Cluster", fontsize="small", loc="best")

    filename_base_cluster = f"umap_clusters_{title_suffix.replace(' ', '_').lower()}"
    save_figure(fig2, filename_base_cluster)
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig2)


# -----------------------
# Driver
# -----------------------

def main():
    # 1) Load all embeddings + metadata
    X, meta = load_all_models()
    y_true = meta["model_name"].values

    # 2) Optional PCA
    if PCA_DIM is not None and PCA_DIM > 0 and PCA_DIM < X.shape[1]:
        X_used, pca = run_pca(X, PCA_DIM)
        title_suffix = f"(PCA {PCA_DIM}D)"
    else:
        X_used = X
        title_suffix = "(orig_dims)"

    # Normalize suffix for filenames (remove parentheses, spaces)
    clean_suffix = title_suffix.replace("(", "").replace(")", "").replace(" ", "_")

    # 3) Cluster and evaluate
    cluster_labels, cluster_summary_df = cluster_and_evaluate(
        X_used,
        labels_true=y_true,
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
    )

    print("\nCluster summary table:")
    print(cluster_summary_df)

    # 4) Visualization (saved to files)
    visualize_2d(X_used, y_true, cluster_labels, title_suffix=clean_suffix)


if __name__ == "__main__":
    main()