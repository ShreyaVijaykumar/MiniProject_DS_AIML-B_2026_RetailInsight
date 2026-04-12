"""
preprocessing.py
----------------
RetailInsight: Customer Purchase Pattern Analysis
SRM Institute of Science and Technology — Mini Project DS AIML-B 2026

OWNER: Team Leader
PURPOSE: Load raw Online Retail II dataset (CSV), clean it, engineer features,
         and save processed output to dataset/processed_data/.
"""

import os
import pandas as pd
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE_DIR, "dataset", "raw_data", "online_retail_II.csv")
PROCESSED_DIR  = os.path.join(BASE_DIR, "dataset", "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─── 1. Load ──────────────────────────────────────────────────────────────────
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load Online Retail II dataset (CSV version).
    """
    print(f"[INFO] Loading raw data from: {path}")
    try:
        df = pd.read_csv(path, encoding="latin1")
        print(f"[INFO] Loaded {len(df):,} rows.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {path}\n"
            f"Please place 'online_retail_II.csv' inside:\n"
            f"{os.path.dirname(path)}\n"
        )


# ─── 2. Basic Cleaning ────────────────────────────────────────────────────────
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, drop nulls, remove cancellations and bad rows.
    """
    print("[INFO] Running basic cleaning...")

    # Standardise column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=["Customer_ID", "Description"])
    print(f"[INFO] Dropped {before - len(df):,} rows with null Customer_ID / Description.")

    # Remove cancellations (Invoice starts with 'C')
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    print(f"[INFO] Removed {before - len(df):,} cancellation rows.")

    # Remove rows with non-positive Quantity or Price
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    print(f"[INFO] Removed {before - len(df):,} invalid Quantity/Price rows.")

    # Clean string columns
    df["Description"] = df["Description"].str.strip().str.upper()
    df["Country"]     = df["Country"].str.strip()

    # Convert Customer_ID to integer
    df["Customer_ID"] = df["Customer_ID"].astype(int)

    print(f"[INFO] Clean dataset: {len(df):,} rows remaining.")
    return df.reset_index(drop=True)


# ─── 3. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived columns used in EDA, visualization, and modelling.
    """
    print("[INFO] Engineering features...")

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Monetary value
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    # Time features
    df["Year"]      = df["InvoiceDate"].dt.year
    df["Month"]     = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"]      = df["InvoiceDate"].dt.hour
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    print("[INFO] Features added successfully.")
    return df


# ─── 4. Save Processed Data ───────────────────────────────────────────────────
def save_processed(df: pd.DataFrame) -> None:
    """
    Save the cleaned dataset.
    """
    out_path = os.path.join(PROCESSED_DIR, "retail_clean.csv")
    df.to_csv(out_path, index=False)

    print(f"[INFO] Processed data saved to: {out_path}")
    print(f"[INFO] Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns.")


# ─── 5. Summary ───────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary."""
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)

    print(f"  Rows         : {df.shape[0]:>12,}")
    print(f"  Columns      : {df.shape[1]:>12}")
    print(f"  Customers    : {df['Customer_ID'].nunique():>12,}")
    print(f"  Invoices     : {df['Invoice'].nunique():>12,}")
    print(f"  Products     : {df['StockCode'].nunique():>12,}")
    print(f"  Countries    : {df['Country'].nunique():>12}")
    print(f"  Date Range   : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    print(f"  Total Revenue: £{df['TotalPrice'].sum():>14,.2f}")

    print("=" * 55 + "\n")


# ─── Pipeline Entry Point ─────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    """Run full preprocessing pipeline."""
    df = load_data()
    df = basic_clean(df)
    df = engineer_features(df)
    print_summary(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    run_pipeline()