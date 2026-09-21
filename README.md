# 🛍️ Mall Customer Segmentation (K-Means)

Grouping mall customers into meaningful segments based on their **annual income** and **spending score**, so a marketing team could treat each group differently.

This repository contains **two versions of the same project**, built one year apart, to show how my approach to a data science problem evolved.

| Version | File | Context |
|---|---|---|
| **v1 (2025)** | [`mall_customer_project_2025.ipynb`](mall_customer_project_2025.ipynb) | My first end-to-end project, built during the *Python & AI* course (36 hours) |
| **v2 (2026)** | [`mall_customer_project_improved_2026.ipynb`](mall_customer_project_improved_2026.ipynb) | A rework of the same project with better methodology, validation, and business interpretation |

## Dataset

`Mall_Customers.csv`: 200 customers with `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The dataset is publicly available on Kaggle.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn (KMeans, Agglomerative Clustering, Gaussian Mixture, Pipeline), joblib.

---

## 🔄 What changed between v1 (2025) and v2 (2026)

| Area | v1 (2025) | v2 (2026) |
|---|---|---|
| **Problem framing** | Followed the course steps in order (including a supervised train/test split) | Framed as an unsupervised segmentation problem with a clear business goal; no train/test split because there is no target |
| **Outliers** | Capped income at the IQR fences automatically | Inspected first: only 2 customers (137k$) were above the fence, they are real high earners, so they were **kept** |
| **Encoding** | One-hot encoded `Age` (turning each age into its own column) | Kept `Age` numeric; `Gender` as a simple 0/1 column used only to *profile* segments |
| **Feature selection** | Clustering runs on all numeric columns (including `CustomerID`), then on two features without scaling | Clustered on `Income` and `Spending` only, **scaled with `StandardScaler`** |
| **Choosing k** | Elbow method only, then `k = 3` | Elbow **+ silhouette score** → `k = 5` (silhouette = **0.555**) |
| **Validation** | None | Stability across 10 random seeds (min ARI = **1.0**) and comparison with Agglomerative and Gaussian Mixture (ARI ≈ 0.94–0.96) |
| **Extra model** | A Random Forest classifier trying to predict the exact spending score (accuracy ≈ 12.5%, since it was treated as 100 separate classes) | Removed: it doesn't fit the problem |
| **Interpretation** | Cluster numbers only | Named segments, profiled by size, age, and gender, with suggested marketing actions |
| **Deployment** | Saved a raw KMeans model and took inputs via `input()` | A scikit-learn `Pipeline` (scaler + KMeans) saved with joblib, plus a validated `predict_segment()` function |
| **Code quality** | Repeated code and reloaded the CSV several times | One clean workflow, fixed random seed, and sanity checks (`assert`) |

## 📊 Results (v2)

Five customer segments were found:

| Segment | Profile | Share of customers |
|---|---|---|
| **Premium Spenders** | High income, high spending | ~19.5% |
| **Frugal High-Earners** | High income, low spending | ~17.5% |
| **Budget Enthusiasts** | Low income, high spending, youngest group | ~11% |
| **Budget Savers** | Low income, low spending, oldest group | ~11.5% |
| **Mainstream** | Middle income, middle spending | ~40.5% |

**Key findings**
- Elbow and silhouette both point to `k = 5`, and the grouping is stable across random seeds.
- Gender does not drive the segments; income and spending behavior do. Age helps *describe* them (Budget Enthusiasts are the youngest, Budget Savers the oldest).
- *Frugal High-Earners* are the biggest growth opportunity, and *Premium Spenders* are the group to protect.

## ⚠️ Limitations

- Only 200 customers, and the spending score is a single number assigned by the mall. A real project would use transaction history (recency, frequency, monetary value).
- Clustering uses two variables only.
- Silhouette measures geometric separation, not business value, so the segments should be validated with the marketing team.

## 🚀 Next steps

- Add behavioral features and try 3-D clustering with `Age`.
- Wrap `predict_segment` in a small Streamlit app.

## ▶️ How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Put `Mall_Customers.csv` in the same folder as the notebooks, then run them with Jupyter.

---

## 💡 What I learned

Rebuilding this project taught me that a good result is not only about running an algorithm. It's about choosing the right features, checking assumptions before changing the data, validating the outcome, and explaining it in terms a business can act on.
