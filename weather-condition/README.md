# Weather Condition Analysis & Linear Regression Model

This project analyzes historical weather data collected from stations around the world and builds a **Linear Regression** model to predict mean temperature.

## 📁 Project Structure

```
weather-condition/
├── data/
│   ├── Summary of Weather.csv        # Main weather dataset (119,040 records, 31 columns)
│   └── Weather Station Locations.csv # Station locations dataset (161 stations, 8 columns)
└── notebooks/
    └── weather_analysis_model.ipynb  # Main analysis and modeling notebook
```

## 📊 Dataset

> 🔗 **Source:** [Kaggle — Weather in World War II (smid80)](https://www.kaggle.com/datasets/smid80/weatherww2)

| File | Rows | Columns | Description |
|---|---|---|---|
| Summary of Weather.csv | 119,040 | 31 | Daily weather measurements per station |
| Weather Station Locations.csv | 161 | 8 | Station coordinates and country information |

**Key variables:** `MaxTemp`, `MinTemp`, `MeanTemp`, `Precip`, `Snowfall`, `WindGustSpd`, etc.

> **Note:** The dataset contains significant missing value issues. Columns such as `WindGustSpd`, `SPD`, and `DR` are missing 99%+ of their values, while `FT`, `FB`, `FTI`, `ITH`, `SD3`, `RHX`, `RHN`, `RVG`, and `WTE` are **entirely empty** and are excluded from the analysis.

## 🔬 Notebook Contents

| Section | Description |
|---|---|
| **1. Imports** | pandas, NumPy, Matplotlib, Seaborn, scikit-learn |
| **2. Data Loading** | Loading both CSV files |
| **3.1. EDA – Stations** | Distribution of 161 stations across 64 countries, statistical summary |
| **3.2. EDA – Weather** | Column types, missing value analysis |
| **4. Visualization** | Bar chart by country, temperature histograms, correlation heatmap |
| **5. Data Preprocessing** | Type conversion for `Precip`/`Snowfall`, dropping rows with missing values in model columns |
| **6. Model – Linear Regression** | `MaxTemp` + `MinTemp` → `MeanTemp` prediction |
| **7. Evaluation** | R², MAE, RMSE metrics |

## 🤖 Model

- **Algorithm:** scikit-learn `LinearRegression`
- **Features (X):** `MaxTemp`, `MinTemp`
- **Target (y):** `MeanTemp`
- **Split:** Train / Test (`train_test_split`)

## ⚙️ Dependencies

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

## 🚀 Running the Notebook

```bash
cd notebooks
jupyter notebook weather_analysis_model.ipynb
```


