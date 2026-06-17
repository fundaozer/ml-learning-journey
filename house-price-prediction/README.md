# House Price Prediction

A machine learning project that predicts house prices using multiple regression models and compares their performance.

## Dataset
- Source: [House Price Regression Dataset](https://www.kaggle.com/datasets/prokshitha/home-value-insights)
- 1000 rows, 8 features
- Key finding: Square_Footage dominates price prediction with 0.99 correlation

## Project Structure
```
├── data/
│   ├── house_price_regression_dataset.csv
│   └── new_df.csv
├── notebooks/
│   ├──eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model.ipynb
└── README.md
```

## Features
| Feature | Description |
|---|---|
| Square_Footage | Size of the house |
| Num_Bedrooms | Number of bedrooms |
| Num_Bathrooms | Number of bathrooms |
| Lot_Size | Size of the lot |
| Garage_Size | Garage capacity |
| Neighborhood_Quality | Quality rating (1-10) |
| Building_Age | Years since construction (derived) |
| Total_Rooms | Bedrooms + Bathrooms (derived) |

## Results

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 8174.58 | 10071.48 | 0.9984 |
| Polynomial Regression | 8311.02 | 10187.32 | 0.9984 |
| Ridge α=0.01 | 8175.22 | 10071.96 | 0.9984 |
| Lasso α=1.0 | 8174.77 | 10071.59 | 0.9984 |
| ElasticNet α=0.01 | 8464.94 | 10337.01 | 0.9983 |

## Conclusion

- All models achieved R² ≈ 0.9984, indicating that Square_Footage alone drives house price prediction with near-perfect accuracy.
- Linear Regression produced the best MAE (8174.58), confirming that the relationship is strongly linear — no complex model needed.
- Polynomial Regression showed no improvement over Linear Regression, meaning there are no significant non-linear patterns in the data.
- Ridge with α=0.01 performed almost identically to Linear Regression, confirming there is no overfitting issue in this dataset.
- Lasso identified Square_Footage as the dominant feature (coef: 249,787) and Neighborhood_Quality as nearly irrelevant (coef: 334).
- ElasticNet with α=0.01 gave the closest result to other models; higher alpha values degraded performance significantly.
- Overall, this dataset is dominated by a single feature. In a real-world scenario, richer data with more correlated features would show
  greater differences between models

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn