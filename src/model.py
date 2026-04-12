"""
model.py
--------
RetailInsight: Customer Purchase Pattern Analysis
SRM Institute of Science and Technology — Mini Project DS AIML-B 2026

OWNER: Teammate B
PURPOSE: Perform RFM (Recency, Frequency, Monetary) analysis and
         K-Means clustering to segment customers into actionable groups.
         Outputs: rfm_table.csv, cluster_profiles.csv, cluster plots.

Requires: preprocessing.py to have been run first (retail_clean.csv must exist).
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed_data")
RESULTS_DIR   = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR    = os.path.join(BASE_DIR, "outputs", "graphs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

CLEAN_CSV     = os.path.join(PROCESSED_DIR, "retail_clean.csv")

# Segment label mapping (assigned after inspecting cluster centroids)
SEGMENT_LABELS = {
    0: "Champions",
    1: "At-Risk",
    2: "Potential Loyalists",
    3: "New Customers",
}

SEGMENT_COLORS = {
    "Champions":          "#f0a500",
    "At-Risk":            "#e05252",
    "Potential Loyalists":"#4fc3f7",
    "New Customers":      "#81c784",
}


# ─── 1. Load Processed Data ───────────────────────────────────────────────────
def load_clean_data() -> pd.DataFrame:
    print(f"[INFO] Loading cleaned data from: {CLEAN_CSV}")
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            "[ERROR] retail_clean.csv not found. Run src/preprocessing.py first."
        )
    df = pd.read_csv(CLEAN_CSV, parse_dates=["InvoiceDate"])
    print(f"[INFO] Loaded {len(df):,} rows.")
    return df


# ─── 2. Build RFM Table ───────────────────────────────────────────────────────
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-customer RFM metrics:
    - Recency   : days since last purchase (lower = better)
    - Frequency : number of unique invoices
    - Monetary  : total spend (£)
    """
    print("[INFO] Computing RFM metrics...")

    # Use day after last transaction as the reference snapshot date
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer_ID")
        .agg(
            Recency  =("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice",     "nunique"),
            Monetary =("TotalPrice",  "sum"),
        )
        .reset_index()
    )

    print(f"[INFO] RFM table built for {len(rfm):,} customers.")
    print(rfm.describe().to_string())
    return rfm


# ─── 3. RFM Scoring (1–5 scale) ──────────────────────────────────────────────
def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign quintile scores (1–5) to each RFM dimension.
    Recency: lower days → higher score (reverse sorted).
    Frequency & Monetary: higher values → higher score.
    """
    print("[INFO] Scoring RFM dimensions (1–5 quintiles)...")

    rfm["R_Score"] = pd.qcut(rfm["Recency"],  q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"]  = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["RFM_Total"]  = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    return rfm


# ─── 4. Elbow + Silhouette to Choose k ───────────────────────────────────────
def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> int:
    """
    Plot elbow curve (inertia) and silhouette scores to choose optimal k.
    """
    print("[INFO] Running elbow method and silhouette analysis...")
    inertias    = []
    sil_scores  = []

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, lbl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#c9d1d9")

    # Elbow
    ax1.plot(list(k_range), inertias, "o-", color="#f0a500", linewidth=2, markersize=7)
    ax1.set_xlabel("Number of Clusters (k)", color="#c9d1d9")
    ax1.set_ylabel("Inertia (WCSS)",          color="#c9d1d9")
    ax1.set_title("Elbow Method",             color="#f0f6fc", fontsize=13)
    ax1.grid(color="#30363d", linestyle="--", linewidth=0.5)

    # Silhouette
    ax2.plot(list(k_range), sil_scores, "s-", color="#4fc3f7", linewidth=2, markersize=7)
    ax2.set_xlabel("Number of Clusters (k)",  color="#c9d1d9")
    ax2.set_ylabel("Silhouette Score",         color="#c9d1d9")
    ax2.set_title("Silhouette Score",          color="#f0f6fc", fontsize=13)
    ax2.grid(color="#30363d", linestyle="--", linewidth=0.5)

    plt.suptitle("Optimal k Selection", color="#f0f6fc", fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "elbow_silhouette.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved: {out}")

    # Return k with highest silhouette score
    optimal_k = list(k_range)[np.argmax(sil_scores)]
    print(f"[INFO] Optimal k by silhouette: {optimal_k}")
    return optimal_k


# ─── 5. K-Means Clustering ───────────────────────────────────────────────────
def run_kmeans(rfm: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Scale R, F, M and run K-Means. Attach cluster labels to rfm DataFrame.
    Note: Recency is negated before scaling so that lower recency = 'better'
    aligns positively with higher F and M in the feature space.
    """
    print(f"[INFO] Running K-Means with k={k}...")

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Recency"] = -features["Recency"]   # negate so all dims: higher = better

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)

    # ── Assign human-readable segment labels ──────────────────────────────────
    # Rank clusters by mean RFM_Total (descending) to assign label order
    cluster_means = rfm.groupby("Cluster")["RFM_Total"].mean().sort_values(ascending=False)
    rank_map      = {cluster_id: rank for rank, cluster_id in enumerate(cluster_means.index)}
    rfm["Segment"] = rfm["Cluster"].map(rank_map).map(SEGMENT_LABELS)

    print("[INFO] Cluster assignment complete.")
    return rfm, X_scaled, scaler


# ─── 6. Cluster Profiles ─────────────────────────────────────────────────────
def build_cluster_profiles(rfm: pd.DataFrame) -> pd.DataFrame:
    profile = (
        rfm.groupby("Segment")
        .agg(
            Customers=("Customer_ID", "count"),
            Avg_Recency  =("Recency",   "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary =("Monetary",  "mean"),
            Total_Revenue=("Monetary",  "sum"),
        )
        .round(2)
        .reset_index()
    )
    out = os.path.join(RESULTS_DIR, "cluster_profiles.csv")
    profile.to_csv(out, index=False)
    print(f"\n[INFO] Cluster profiles saved: {out}")
    print(profile.to_string(index=False))
    return profile


# ─── 7. Visualisations ───────────────────────────────────────────────────────
def plot_rfm_distributions(rfm: pd.DataFrame) -> None:
    """Histograms of R, F, M before clustering."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    cols  = ["Recency", "Frequency", "Monetary"]
    clrs  = ["#f0a500", "#4fc3f7", "#81c784"]
    for ax, col, clr in zip(axes, cols, clrs):
        ax.set_facecolor("#0f1117")
        ax.hist(rfm[col], bins=50, color=clr, alpha=0.85, edgecolor="none")
        ax.set_title(col, color="#f0f6fc", fontsize=13)
        ax.set_xlabel("Value", color="#c9d1d9")
        ax.set_ylabel("Count", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.suptitle("RFM Distributions", color="#f0f6fc", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "rfm_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved: {out}")


def plot_cluster_scatter_pca(rfm: pd.DataFrame, X_scaled: np.ndarray) -> None:
    """PCA 2D scatter plot of customer clusters."""
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    segments = rfm["Segment"].values
    for seg in SEGMENT_LABELS.values():
        mask = segments == seg
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=SEGMENT_COLORS[seg], label=seg,
            alpha=0.65, s=18, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", color="#c9d1d9")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", color="#c9d1d9")
    ax.set_title("Customer Segments — PCA Projection", color="#f0f6fc", fontsize=14, pad=12)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.5)
    legend = ax.legend(frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("#161b22")
    legend.get_frame().set_edgecolor("#30363d")
    for text in legend.get_texts():
        text.set_color("#c9d1d9")

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "cluster_pca_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved: {out}")


def plot_segment_pie(rfm: pd.DataFrame) -> None:
    """Pie chart of customer count per segment."""
    counts  = rfm["Segment"].value_counts()
    colors  = [SEGMENT_COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 2},
        textprops={"color": "#f0f6fc", "fontsize": 11},
    )
    for at in autotexts:
        at.set_color("#0f1117")
        at.set_fontweight("bold")

    ax.set_title("Customer Segment Distribution", color="#f0f6fc", fontsize=14, pad=14)
    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "segment_pie_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved: {out}")


def plot_rfm_radar(profile: pd.DataFrame) -> None:
    """
    Radar (spider) chart comparing normalised avg R, F, M across segments.
    """
    dims = ["Avg_Recency", "Avg_Frequency", "Avg_Monetary"]
    labels = ["Recency\n(lower=better)", "Frequency", "Monetary (£)"]
    N = len(dims)

    # Normalise each dimension to [0, 1] — invert recency
    norm = profile[dims].copy()
    for d in dims:
        mn, mx = norm[d].min(), norm[d].max()
        norm[d] = (norm[d] - mn) / (mx - mn) if mx != mn else 0
    norm["Avg_Recency"] = 1 - norm["Avg_Recency"]   # invert: closer = better

    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b22")

    for _, row in profile.iterrows():
        seg   = row["Segment"]
        vals  = norm.loc[norm.index == row.name, dims].values.flatten().tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=SEGMENT_COLORS[seg], linewidth=2, label=seg)
        ax.fill(angles, vals, color=SEGMENT_COLORS[seg], alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color="#c9d1d9", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], color="#606060", fontsize=8)
    ax.grid(color="#30363d", linestyle="--", linewidth=0.6)
    ax.spines["polar"].set_edgecolor("#30363d")
    ax.set_title("Segment RFM Profiles (normalised)", color="#f0f6fc", fontsize=13, pad=20)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    legend.get_frame().set_facecolor("#161b22")
    legend.get_frame().set_edgecolor("#30363d")
    for text in legend.get_texts():
        text.set_color("#c9d1d9")

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "segment_radar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved: {out}")


# ─── 8. Save Full RFM Table ───────────────────────────────────────────────────
def save_rfm_table(rfm: pd.DataFrame) -> None:
    out = os.path.join(RESULTS_DIR, "rfm_table.csv")
    rfm.to_csv(out, index=False)
    print(f"[INFO] Full RFM table saved: {out}")


# ─── Pipeline Entry Point ─────────────────────────────────────────────────────
def run_model() -> pd.DataFrame:
    df       = load_clean_data()
    rfm      = build_rfm(df)
    rfm      = score_rfm(rfm)

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Recency"] = -features["Recency"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Uncomment to run elbow analysis (takes ~30s extra):
    # find_optimal_k(X_scaled)

    rfm, X_scaled, scaler = run_kmeans(rfm, k=4)
    profile               = build_cluster_profiles(rfm)

    plot_rfm_distributions(rfm)
    plot_cluster_scatter_pca(rfm, X_scaled)
    plot_segment_pie(rfm)
    plot_rfm_radar(profile)
    save_rfm_table(rfm)

    print("\n[INFO] Modelling complete. All outputs saved to outputs/.")
    return rfm


if __name__ == "__main__":
    run_model()
