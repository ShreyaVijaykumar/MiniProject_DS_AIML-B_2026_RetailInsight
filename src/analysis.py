"""
analysis.py
-----------
RetailInsight: Customer Purchase Pattern Analysis
SRM Institute of Science and Technology — Mini Project DS AIML-B 2026

OWNER: Teammate A
PURPOSE: Perform Association Rule Mining using the Apriori algorithm
         on the cleaned Online Retail II dataset.
         Outputs: frequent itemsets CSV, association rules CSV, top rules plot.

Requires: preprocessing.py to have been run first (retail_clean.csv must exist).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed_data")
RESULTS_DIR   = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR    = os.path.join(BASE_DIR, "outputs", "graphs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

CLEAN_CSV     = os.path.join(PROCESSED_DIR, "retail_clean.csv")


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


# ─── 2. Build Transaction Matrix ─────────────────────────────────────────────
def build_basket(df: pd.DataFrame, country: str = "United Kingdom") -> pd.DataFrame:
    """
    Filter to one country (default UK — largest segment) and build a
    boolean basket matrix: rows = Invoice, columns = product Description,
    values = True/False (was the product in that invoice?).

    We limit to UK because cross-country basket analysis is noisy and the
    UK transactions make up ~90 % of volume.
    """
    print(f"[INFO] Building basket matrix for country: {country}")
    df_c = df[df["Country"] == country].copy()

    # Aggregate — a product can appear in duplicate line items on one invoice
    basket = (
        df_c.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    # Convert quantities to boolean (bought / not bought)
    basket = basket.map(lambda x: True if x > 0 else False)

    # Drop products that appear in fewer than 10 invoices to keep matrix sparse
    basket = basket.loc[:, basket.sum() >= 10]

    print(f"[INFO] Basket matrix shape: {basket.shape[0]:,} invoices × {basket.shape[1]:,} products.")
    return basket


# ─── 3. Apriori — Frequent Itemsets ──────────────────────────────────────────
def find_frequent_itemsets(basket: pd.DataFrame, min_support: float = 0.02) -> pd.DataFrame:
    """
    Run the Apriori algorithm.
    min_support = 0.02 means the itemset appears in at least 2 % of invoices.
    Lower this value (e.g. 0.01) to get more itemsets — but runtime increases.
    """
    print(f"[INFO] Running Apriori (min_support={min_support})...")
    frequent_items = apriori(basket, min_support=min_support, use_colnames=True)
    frequent_items["length"] = frequent_items["itemsets"].apply(len)
    frequent_items = frequent_items.sort_values("support", ascending=False)
    print(f"[INFO] Found {len(frequent_items):,} frequent itemsets.")

    out_path = os.path.join(RESULTS_DIR, "frequent_itemsets.csv")
    frequent_items.to_csv(out_path, index=False)
    print(f"[INFO] Saved: {out_path}")
    return frequent_items


# ─── 4. Association Rules ─────────────────────────────────────────────────────
def generate_rules(
    frequent_items: pd.DataFrame,
    metric: str       = "lift",
    min_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.

    Key metrics:
    - Support    : P(A ∩ B)          — how often the pair appears
    - Confidence : P(B | A)          — if A, how likely is B
    - Lift       : confidence / P(B) — how much better than random
    Lift > 1 = positive association; lift >> 1 = strong rule.
    """
    print(f"[INFO] Generating association rules (metric={metric}, min={min_threshold})...")
    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Convert frozensets to readable strings for CSV export
    rules_export = rules.copy()
    rules_export["antecedents"] = rules_export["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules_export["consequents"] = rules_export["consequents"].apply(lambda x: ", ".join(list(x)))

    out_path = os.path.join(RESULTS_DIR, "association_rules.csv")
    rules_export.to_csv(out_path, index=False)
    print(f"[INFO] Found {len(rules):,} rules. Saved: {out_path}")
    return rules


# ─── 5. Visualise Top Rules ───────────────────────────────────────────────────
def plot_top_rules(rules: pd.DataFrame, top_n: int = 15) -> None:
    """
    Bar chart of the top N rules by lift, with confidence shown as colour.
    """
    top = rules.head(top_n).copy()
    top["rule"] = (
        top["antecedents"].apply(lambda x: ", ".join(list(x)).title()[:30])
        + "  →  "
        + top["consequents"].apply(lambda x: ", ".join(list(x)).title()[:25])
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    colors = plt.cm.YlOrRd(top["confidence"].values)
    bars   = ax.barh(top["rule"][::-1], top["lift"][::-1], color=colors[::-1], height=0.65)

    # Colorbar for confidence
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(top["confidence"].min(), top["confidence"].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Confidence", color="#c9d1d9", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

    # Labels inside bars
    for bar, val in zip(bars, top["lift"][::-1]):
        ax.text(
            bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="right", fontsize=9, color="#0f1117", fontweight="bold"
        )

    ax.set_xlabel("Lift", color="#c9d1d9", fontsize=11)
    ax.set_title(f"Top {top_n} Association Rules by Lift", color="#f0f6fc", fontsize=14, pad=14)
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    ax.xaxis.label.set_color("#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="x", color="#30363d", linestyle="--", linewidth=0.6)
    ax.set_xlim(0, top["lift"].max() * 1.12)

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "top_association_rules.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


def plot_support_confidence_scatter(rules: pd.DataFrame) -> None:
    """
    Scatter plot: Support vs Confidence, point size = Lift.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    sc = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        s=rules["lift"] * 18,
        cmap="plasma",
        alpha=0.75,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Lift", color="#c9d1d9", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")

    ax.set_xlabel("Support",    color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Confidence", color="#c9d1d9", fontsize=11)
    ax.set_title("Support vs Confidence (size = Lift)", color="#f0f6fc", fontsize=13, pad=12)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "support_confidence_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")


# ─── 6. Print Top Rules Summary ───────────────────────────────────────────────
def print_top_rules(rules: pd.DataFrame, n: int = 10) -> None:
    print("\n" + "=" * 60)
    print(f"  TOP {n} ASSOCIATION RULES (by Lift)")
    print("=" * 60)
    for i, row in rules.head(n).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        print(f"  {i+1:>2}. {ant[:35]:<35}  →  {con[:30]}")
        print(f"      Support: {row['support']:.4f}  |  Confidence: {row['confidence']:.4f}  |  Lift: {row['lift']:.4f}")
    print("=" * 60 + "\n")


# ─── Pipeline Entry Point ─────────────────────────────────────────────────────
def run_analysis() -> dict:
    df             = load_clean_data()
    basket         = build_basket(df)
    frequent_items = find_frequent_itemsets(basket)
    rules          = generate_rules(frequent_items)
    print_top_rules(rules)
    plot_top_rules(rules)
    plot_support_confidence_scatter(rules)
    print("[INFO] Association rule mining complete.")
    return {"frequent_items": frequent_items, "rules": rules}


if __name__ == "__main__":
    run_analysis()