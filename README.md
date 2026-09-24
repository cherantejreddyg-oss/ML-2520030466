# ML-S6-2520030466: Accident Hotspot Prediction Using Random Forest and Spatial Clustering

A terminal-based Machine Learning application designed for analyzing road accident black-spots in Hyderabad City, predicting location risk levels, identifying high-danger hotspots, and performing spatial clustering analysis.

---

## 📌 Project Overview

Road accidents are a critical public safety challenge. This project leverages historical accident black-spot data from Hyderabad City (2023–2025) to:
1. **Preprocess and Clean** complex multi-row header raw CSV datasets.
2. **Engineer Key Risk Metrics** (Severity Score, Fatality Rate, Yearly Growth, Average Yearly Accidents).
3. **Classify Location Risk** into `LOW`, `MEDIUM`, and `HIGH` using a **Random Forest Classifier**.
4. **Identify & Rank Hotspots** based on quantitative severity statistics.
5. **Perform Analytical Spatial Clustering** using **K-Means Clustering** as a feature-space proxy for spatial grouping.

---

## 📁 Project Structure

```
accident_hotspot_prediction/
│
├── main.py                                      # Main interactive terminal application entrypoint
├── data/
│   └── AccidentProneAreas_Complete(Complete Dataset) (3) (1).csv  # Real Hyderabad accident dataset
│
├── src/
│   ├── __init__.py                              # Package initializer
│   ├── data_loader.py                           # Raw CSV loader and banner header stripper
│   ├── preprocessing.py                         # Data cleaning, feature engineering, and risk labeling
│   ├── eda.py                                   # Text-based ASCII tables and visual chart generation
│   ├── random_forest_model.py                   # Random Forest Classifier training & evaluation
│   ├── clustering.py                            # Spatial coordinate audit & KMeans clustering proxy
│   ├── prediction.py                            # Interactive risk prediction CLI
│   └── report.py                                # Comprehensive report compiler
│
├── outputs/
│   ├── charts/                                  # Saved EDA visualization plots (.png)
│   └── reports/                                 # Exported project text report (.txt)
│
├── requirements.txt                             # Required Python packages
└── README.md                                    # Comprehensive project documentation
```

---

## ⚙️ Installation & Requirements

### 1. Prerequisites
- Python 3.10+
- `pip` package manager

### 2. Install Dependencies
Run the following command in your terminal:

```bash
py -m pip install -r requirements.txt
```
*(Or `pip install -r requirements.txt` depending on your OS configuration)*

---

## 🚀 How to Run the Application

Navigate to the project directory and execute:

```bash
py main.py
```
*(Or `python main.py`)*

---

## 🖥️ Application Interactive Menu

When you run `main.py`, the following interactive terminal menu is presented:

```
==================================================
      ACCIDENT HOTSPOT PREDICTION SYSTEM
==================================================
 1. Load Dataset
 2. View Dataset Summary
 3. Data Preprocessing & Feature Engineering
 4. Exploratory Data Analysis (EDA)
 5. Train Random Forest Model
 6. Predict Accident Risk
 7. Perform Spatial Clustering
 8. View Accident Hotspots
 9. Model Evaluation & Feature Importance
10. Generate Complete Project Report
11. Exit
==================================================
```

---

## 📊 Summary of Modules & Methodologies

1. **`data_loader.py`**: Reads raw CSV data, strips title headers and metadata, handles missing files gracefully.
2. **`preprocessing.py`**: Trims text whitespace, standardizes jurisdiction names, converts data types, fills NaNs with 0, and calculates engineered metrics. Uses an explicit rule:
   $$\text{Severity Score} = \text{Total Accidents} + (1.5 \times \text{Total Fatalities})$$
   - `LOW` Risk: $\text{Severity Score} \le 13.0$
   - `MEDIUM` Risk: $13.0 < \text{Severity Score} \le 18.0$
   - `HIGH` Risk: $\text{Severity Score} > 18.0$
3. **`eda.py`**: Displays ASCII terminal tables and exports high-resolution visual plots to `outputs/charts/`.
4. **`random_forest_model.py`**: Trains `RandomForestClassifier(n_estimators=100)` with `train_test_split(test_size=0.3, random_state=42)`. Evaluates Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Feature Importances.
5. **`prediction.py`**: Provides interactive prediction for existing blackspots or custom feature inputs.
6. **`clustering.py`**: Audits geographic coordinates (latitude/longitude) and applies `KMeans` analytical clustering on normalized severity features.
7. **`report.py`**: Compiles and exports the complete project report to `outputs/reports/accident_prediction_report.txt`.
