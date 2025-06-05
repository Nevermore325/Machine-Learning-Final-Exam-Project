# House Price Prediction – ML Final Exam Project

**Course:** Machine Learning  
**Student:** Ruben Hambardzumyan  
**Date:** June 2025  

## Overview

This project builds and evaluates several regression models to predict the selling price (`SalePrice`) of houses in Ames, Iowa.  
Starting with a baseline **Linear Regression**, we progressively add preprocessing, feature engineering and alternative algorithms (Ridge, Lasso, ElasticNet, DecisionTree, GradientBoosting) to measure performance gains.

## Dataset

* **Name:** *Ames Housing Dataset*  
* **Size:** 2 080 observations, 81 raw features  
* **Source:** Kaggle – downloaded automatically with `kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")` in the notebook.

## Quick Start

```bash
# Clone the repo and after that
# Launch the notebook
jupyter lab Machine_Learning_Final_Project-3.ipynb
```

> **Tip:** You need a Kaggle API token (`~/.kaggle/kaggle.json`) for the data‑download cell to work.

## Project Workflow

1. **Data Loading & Initial EDA** – preview, missing‑value summary, correlation matrix.  
2. **Data Cleaning** – imputations + one‑hot encoding of categorical variables.  
3. **Scaling** – `StandardScaler` for numeric features (alternative `MinMaxScaler` for engineered set).  
4. **Baseline Model** – Linear Regression *(MAE ≈ 16.3 k, RMSE ≈ 29.2 k, R² ≈ 0.8935)*.  
5. **Feature Engineering** –  
   • `TotalSF` = 1st + 2nd floor + basement  
   • `TotalBath` (½‑baths counted as 0.5)  
   • `HouseAge` (2025 – YearBuilt)  
   • `Remodeled` indicator  
6. **Model Zoo** – Ridge, Lasso, ElasticNet, DecisionTree, GradientBoosting (models compared on the same 80 / 20 split).  
7. **Visual Diagnostics** – actual vs predicted scatter plots.  
8. **Next Steps** – add Random Forest and XGBoost, hyper‑parameter tuning & k‑fold CV.

## Preliminary Results

* **Baseline Linear Regression:** RMSE ≈ 29 000 $; R² ≈ 0.894.  
* **Engineered + Scaled features:** similar performance (see notebook).  
* **Best of current alternative models:** Gradient Boosting performed best among the tested tree‑based methods. See the comparison cell for exact scores.

## File Structure

```
├── Machine_Learning_Final_Project-3.ipynb  ← Jupyter notebook with all code & plots
├── README.md                               ← **you are here**
└── data/                                   ← downloaded CSVs (auto‑generated)
```

## Requirements

* Python ≥ 3.9  
* pandas, numpy, scikit‑learn, matplotlib, seaborn  
* plotly, kagglehub  
* (optional) xgboost for future experiments

## Reproducibility

All random processes use `random_state=42` to enable deterministic splits and model behaviour.
