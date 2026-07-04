# Iris Species Classification Project

This project aims to classify Iris flowers into their respective species (Iris-setosa, Iris-versicolor, Iris-virginica) based on sepal and petal measurements (length and width) using the famous **Iris dataset**. The project applies Exploratory Data Analysis (EDA), data preprocessing, model training, and hyperparameter optimization.

---

## About the Dataset

The dataset contains a total of **150 samples**, and the following features are measured for each sample:
* **SepalLengthCm** (Sepal Length)
* **SepalWidthCm** (Sepal Width)
* **PetalLengthCm** (Petal Length)
* **PetalWidthCm** (Petal Width)

The target variable **Species** consists of 3 classes:
1. `Iris-setosa` (0)
2. `Iris-versicolor` (1)
3. `Iris-virginica` (2)

---

## Project Workflow and Implemented Steps

### 1. Exploratory Data Analysis (EDA)
Various plots were generated to analyze the distribution of data and the separability of classes:
* **Pairplot**: The distribution of pairwise feature combinations is displayed, color-coded by class (`hue`).
* **Correlation Matrix (Heatmap)**: The relationships between features are numerically visualized.
* **Boxplots**: The distribution ranges of each feature across different flower species are shown to identify distinctive characteristics.

### 2. Data Preprocessing
* The `Id` column was removed from the dataset since it does not contribute to the analysis.
* Target classes (`Species`) were encoded into numerical values using **LabelEncoder**.
* Features were standardized using **StandardScaler** to improve model performance.
* The dataset was split into 75% Training and 25% Testing sets.

### 3. Model Training and Hyperparameter Tuning
Three different machine learning algorithms were trained and their performances were compared:

| Model | Optimization Method | Tuned Hyperparameters |
| :--- | :--- | :--- |
| **Gaussian Naive Bayes** | Default | No parameter tuning was performed. |
| **Logistic Regression** | `GridSearchCV` | `C` (Regularization strength), `penalty`, and `solver` |
| **Support Vector Classifier (SVC)** | `GridSearchCV` | `C`, `kernel`, and `gamma` |

---

## How to Run?

You can follow these steps to run the notebook file:

1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Run the [iris_species_model.ipynb](notebook/iris_species_model.ipynb) file located in the `notebook/` directory.
