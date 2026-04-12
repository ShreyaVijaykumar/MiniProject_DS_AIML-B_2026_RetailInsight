# RetailInsight: Customer Purchase Pattern Analysis Using Data Mining

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Abstract

Retail businesses generate massive volumes of transactional data daily, yet most of this data remains underutilized in driving strategic decisions. This project, **RetailInsight**, applies data mining techniques to the UCI Online Retail II dataset — a real-world e-commerce transaction log spanning 2009–2011 from a UK-based retailer. Using Python-based data science tools, we perform end-to-end analysis: cleaning raw invoice data, engineering customer-level features, discovering frequent itemsets via the Apriori algorithm, segmenting customers through RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering, and visualizing actionable insights. The project uncovers hidden purchasing patterns such as product affinity pairs, seasonal demand trends, and high-value customer segments. Findings can directly inform inventory management, personalized marketing campaigns, and cross-selling strategies. All code is modular, reproducible, and documented following academic and industry standards. This work aligns with SDG Goal 9 by fostering data-driven innovation in retail infrastructure.

---

## Problem Statement

Understanding customer purchasing behavior helps businesses design better marketing strategies, reduce churn, and optimize product placement. This project analyzes retail transaction data to discover:
- Which products are frequently bought together (Association Rule Mining)
- How customers can be segmented based on buying behavior (RFM + Clustering)
- What seasonal trends exist in purchasing patterns (Time-Series EDA)

---

## Dataset Source

| Field | Detail |
|---|---|
| **Name** | Online Retail II |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) |
| **Kaggle Mirror** | [Kaggle - Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) |
| **Size** | ~1M rows, 8 columns |
| **Period** | Dec 2009 – Dec 2011 |
| **Description** | Transactional data for a UK-based non-store online retail company selling unique all-occasion giftware |

**Columns:**
- `InvoiceNo` — Invoice number (prefix 'C' = cancellation)
- `StockCode` — Product code
- `Description` — Product name
- `Quantity` — Units per transaction
- `InvoiceDate` — Date and time of invoice
- `UnitPrice` — Price per unit (GBP)
- `CustomerID` — Unique customer identifier
- `Country` — Customer's country

---

## Methodology / Workflow

```
1. Problem Identification
        ↓
2. Dataset Collection  (Kaggle / UCI - Online Retail II)
        ↓
3. Data Cleaning & Preprocessing
   - Remove cancellations, nulls, negative quantities
   - Feature engineering: TotalPrice, Month, DayOfWeek
        ↓
4. Exploratory Data Analysis
   - Top products, revenue by country, monthly sales trends
        ↓
5. Data Visualization
   - Bar charts, heatmaps, word clouds, time-series plots
        ↓
6. Model Development
   - Association Rule Mining (Apriori + mlxtend)
   - RFM Analysis + K-Means Clustering
        ↓
7. Result Interpretation
   - Business recommendations from cluster profiles
```

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Static visualizations |
| Seaborn | Statistical visualizations |
| Scikit-learn | K-Means clustering, preprocessing |
| mlxtend | Apriori algorithm and association rules |
| WordCloud | Product description word clouds |
| Jupyter Notebook | Interactive analysis environment |
| GitHub | Version control and collaboration |

---

## Results / Findings

1. **Top Revenue Products**: "REGENCY CAKESTAND 3 TIER" and "WHITE HANGING HEART T-LIGHT HOLDER" account for a disproportionate share of revenue — ideal for featured placement.

2. **Peak Purchasing Period**: November–December shows ~40% spike in transaction volume — confirming holiday-driven demand. Inventory should be optimized by October.

3. **Association Rules**: Products in the "LUNCH BAG" category show strong lift (>3.0) when purchased alongside stationery items — cross-sell opportunity.

4. **Customer Segments (K-Means, k=4)**:
   - **Champions** — High RFM score; recently active, frequent, high spend
   - **At-Risk** — Previously high spenders, not recently active → re-engagement campaigns
   - **Potential Loyalists** — Moderate frequency, growing spend
   - **New Customers** — Low frequency, recent first purchase

---

## Repository Structure

```
MiniProject/
├── README.md
├── requirements.txt
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
├── dataset/
│   ├── raw_data/          ← Place online_retail_II.xlsx here
│   └── processed_data/    ← Auto-generated by preprocessing pipeline
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   └── visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   └── model.py
├── outputs/
│   ├── graphs/            ← Auto-saved plots
│   └── results/           ← CSV outputs (RFM table, rules, clusters)
└── report/
    └── mini_project_report.pdf
```

---

## SDG Alignment

**Goal 9: Industry, Innovation and Infrastructure**

This project demonstrates how data-driven analytics can modernize retail operations — reducing waste through demand forecasting, enabling precision marketing through customer segmentation, and supporting evidence-based decisions in supply chain management. Making this analysis open-source and reproducible lowers the barrier for small businesses to adopt data science practices.

---

## Team Members

| Name | Role | Contribution |
|---|---|---|
| [Your Name] | Team Leader | Data collection, preprocessing pipeline, EDA, model development, report |
| [Teammate A] | Member | Association rule mining (analysis.py), visualization notebook, graphs |
| [Teammate B] | Member | RFM analysis & clustering (model.py), results interpretation, presentation |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/MiniProject_DS_AIML-B_2026_RetailInsight.git
cd MiniProject_DS_AIML-B_2026_RetailInsight

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in dataset/raw_data/
#    https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

# 4. Run preprocessing
python src/preprocessing.py

# 5. Run analysis
python src/analysis.py

# 6. Run model
python src/model.py

# 7. Or explore interactively via Jupyter
jupyter notebook notebooks/
```

---

*SRM Institute of Science and Technology — Mini Project | Data Science | AIML-B 2026*
