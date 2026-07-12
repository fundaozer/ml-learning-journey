# Medical Cost Personal Datasets Regression Models

This project analyzes personal medical information to predict medical insurance costs (`charges`) using various regression models. It performs Exploratory Data Analysis (EDA), pre-processes the dataset, and compares multiple regression models tuned with **GridSearchCV**.

## 📁 Project Structure

```
medical-cost/
├── data/
│   └── insurance.csv              # Main insurance dataset (1,338 records, 7 columns)
├── notebook/
│   └── regression_models.ipynb    # Main analysis, preprocessing, and modeling notebook
└── README.md                      # Project documentation
```

## 📊 Dataset

> 🔗 **Source:** [Kaggle — Medical Cost Personal Datasets (mirichoi0218)](https://www.kaggle.com/datasets/mirichoi0218/insurance)

| File | Rows | Columns | Description |
|---|---|---|---|
| insurance.csv | 1,338 | 7 | Medical details and charges for insurance policyholders |

**Key variables:**
- `age`: Age of primary beneficiary
- `sex`: Insurance contractor gender (female, male)
- `bmi`: Body mass index ($kg / m^2$)
- `children`: Number of children covered by health insurance / Number of dependents
- `smoker`: Smoking status (yes, no)
- `region`: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
- `charges`: Individual medical costs billed by health insurance (Target variable)

## 🔬 Notebook Contents

| Section | Description |
|---|---|
| **1. Imports** | numpy, pandas, matplotlib, seaborn, scikit-learn |
| **2. Data Loading** | Loading the `insurance.csv` dataset |
| **3. Exploratory Data Analysis (EDA)** | General stats, value counts for categoric variables, missing value check |
| **4. Feature Preprocessing** | One-hot encoding for `sex`, `smoker`, and `region` using `pd.get_dummies`, standard scaling numerical features |
| **5. Model Tuning & Evaluation** | Defining `models_params` and running `GridSearchCV` with 5-fold cross-validation |
| **6. Results Comparison** | Comparing models in terms of $R^2$, RMSE, and MAE |

## 🤖 Models & Tuning

A total of 8 regression models were built and compared:
- **Linear Regression**
- **Polynomial Regression** (Pipeline: `PolynomialFeatures` -> `StandardScaler` -> `LinearRegression`)
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet Regression**
- **Support Vector Regression (SVR)** (Target scaled using `TransformedTargetRegressor` to handle scale sensitivity)
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regressor**

All hyperparameter tuning was conducted using `GridSearchCV` to optimize the $R^2$ score.

## 🏆 Model Comparison Results

Below is the sorted performance table based on the test set evaluation:

| Model | Best Params | R² | RMSE | MAE |
|---|---|---|---|---|
| **Decision Tree** | `{'max_depth': 5, 'min_samples_leaf': 8, 'min_samples_split': 2}` | **0.900904** | 3777.89 | 2285.75 |
| **SVR** | `{'regressor__C': 1, 'regressor__gamma': 'scale', 'regressor__kernel': 'rbf'}` | **0.885398** | 4062.74 | 2298.61 |
| **Polynomial Regression** | `{'poly__degree': 2}` | **0.874990** | 4243.21 | 2762.13 |
| **KNN** | `{'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}` | **0.859151** | 4504.02 | 2870.09 |
| **Lasso** | `{'alpha': 100}` | **0.787788** | 5528.51 | 4020.13 |
| **Ridge** | `{'alpha': 10}` | **0.787120** | 5537.21 | 4036.41 |
| **ElasticNet** | `{'alpha': 0.01, 'l1_ratio': 0.05}` | **0.787103** | 5537.42 | 4035.80 |
| **Linear Regression** | `No parameter` | **0.786686** | 5542.85 | 4023.65 |

* **Decision Tree** yields the best prediction performance with an $R^2$ score of **90.09%**.
* Scaled **SVR** using target scaling delivers strong performance (**88.54%**), improving significantly from its unscaled baseline.

## ⚙️ Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 🚀 Running the Notebook

```bash
cd notebook
jupyter notebook regression_models.ipynb
```
