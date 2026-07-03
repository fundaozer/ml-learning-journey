# Vehicle Price Prediction & Sellability Classification

This repository contains a comprehensive data science project focused on predicting used vehicle prices and analyzing their sales velocity (sellability). The project employs multiple regression models and a binary classification model.

---

## 1. Project Goal
The primary objectives of this project are:
* **To predict used vehicle prices** using various regression models (Linear, Polynomial, Ridge, Lasso, ElasticNet) and find the most accurate appraiser.
* **To predict vehicle sellability** (whether a car will sell quickly, i.e., in 30 days or less) using a Logistic Regression classifier.
* **To perform a "Good Deal" analysis** by comparing the actual listing prices against the predicted fair market prices to see if underpriced cars actually sell faster.

---

## 2. Dataset Description
The dataset contains historical listings of used cars with the following specifications:
* **name:** The brand and model name of the car.
* **year:** The manufacturing year of the vehicle.
* **selling_price:** The price at which the owner is selling the vehicle (Target for regression).
* **km_driven:** Total kilometers the vehicle has been driven.
* **fuel:** Fuel type (Diesel, Petrol, CNG, LPG).
* **seller_type:** Seller category (Individual, Dealer, Trustmark Dealer).
* **transmission:** Gear transmission type (Manual, Automatic).
* **owner:** Ownership history (First Owner, Second Owner, etc.).
* **mileage(km/ltr/kg):** Fuel efficiency of the vehicle.
* **engine:** Engine displacement capacity in CC.
* **max_power:** Maximum horsepower of the engine.
* **seats:** The number of seats in the vehicle.
* **days_on_market:** The number of days the vehicle stayed listed on the market (Note: This is a **synthetically generated column** created for classification analysis).

---

## 3. Missing Value Analysis
During the initial data check, missing values were discovered in the following features:
* `mileage`: 221 missing values
* `engine`: 221 missing values
* `max_power`: 215 missing values
* `seats`: 221 missing values

Since the missing values constituted a tiny fraction of the dataset (~2.7% of the 8,128 original rows), these rows were dropped using `dropna()` to maintain high data quality. Additionally, selling prices above the 99th percentile were removed as outliers to prevent models from skewing toward ultra-expensive outlier vehicles.

---

## 4. Visualizations
Three primary relations were visualized during the Exploratory Data Analysis (EDA) stage:
* **Price Distribution:** Visualized using a histogram. It showed a heavily right-skewed distribution, highlighting the necessity of removing the top 1% luxury car outliers.
* **Mileage vs. Price:** A scatter plot indicating a negative correlation; as kilometers driven increase, the vehicle price generally drops.
* **Model Year vs. Price:** A scatter plot confirming a strong positive relationship; newer model years fetch significantly higher prices.
* **Average Price by Fuel Type:** A bar chart showing that Diesel cars generally command higher average prices compared to Petrol and alternative fuel vehicles.

---

## 5. Feature Engineering
Several new features were engineered to increase predictive power:
* `car_age`: Chronological age of the car, computed as `2026 - year`.
* `mileage_per_year`: Average yearly usage of the vehicle: `km_driven / (car_age + 1)`.
* `power_per_engine`: Engine efficiency ratio: `max_power / engine`.
* `is_fast_sale`: A binary target class generated from `days_on_market`. It is set to `1` if a car sells in 30 days or less (fast sale) and `0` otherwise.
* `is_good_deal`: A binary indicator set to `1` if the actual price is below the predicted market price, representing an underpriced deal.

---

## 6. Linear Regression Results
A standard OLS Linear Regression model was fitted on the scaled data.
* **MAE (Mean Absolute Error):** 221,647.10
* **RMSE (Root Mean Squared Error):** 364,721.32
* **$R^2$ Score:** 0.728594

**Interpretation:** The base linear model explains approximately 72.8% of the variance in used car prices. An MAE of ~221,647 indicates that, on average, the model's price predictions deviate by about 221k units from the actual price.

---

## 7. Polynomial Regression Results
We implemented a Polynomial Regression model of degree 2 using a scikit-learn pipeline.
* **$R^2$ Score:** 0.919043
* **RMSE:** 199,194.94

### Overfitting Check
To verify if the model was overfitting, we scored the training and test sets separately:
* **Train $R^2$ Score:** 0.920478
* **Test $R^2$ Score:** 0.919043

**Interpretation:** The Polynomial model performs exceptionally well, explaining nearly 92% of the price variance. Because the Train $R^2$ and Test $R^2$ are extremely close (less than 0.15% difference), we can confidently state that the model **did not overfit** and generalizes very well.

---

## 8. Ridge, Lasso, and ElasticNet Comparison
We applied regularization techniques to see if they could improve prediction stability:
* **Ridge Regression:** Evaluated across different alpha values `[0.01, 0.1, 1, 10, 100]`. The best alpha was `0.01` ($R^2 = 0.728592$, RMSE = 364,721.91), which performed nearly identically to OLS Linear Regression.
* **Lasso Regression:** Evaluated with `alpha=1.0` ($R^2 = 0.728592$, RMSE = 364,722.02). Analyzing the coefficients (`coef_df`) showed that Lasso did not reduce any coefficient to exactly 0, indicating all features carried some weight.
* **ElasticNet Regression:** Evaluated across multiple L1 ratios `[0.1, 0.3, 0.5, 0.7, 0.9]` with `alpha=1.0`. The best ratio was `0.9` (closer to Lasso) yielding an $R^2$ of 0.702. Overall, ElasticNet performed the worst because its penalty constraints were too aggressive for this feature set.

---

## 9. Best Regression Model
The **Polynomial Regression (Degree 2)** model is the best performing model. It successfully captures the non-linear relationship between a car's age, mileage, and power efficiency vs. its market value, slashing the baseline RMSE by nearly 45%.

---

## 10. Logistic Regression Results
We modeled vehicle sellability using Logistic Regression. To counter the target class imbalance (5,838 slow sales vs. 1,904 fast sales), we applied `class_weight='balanced'`.
* **Accuracy:** 50.2%
* **Classification Report:**
  * **Class 0 (Hard to sell):** Precision = 0.74, Recall = 0.50, F1-score = 0.60
  * **Class 1 (Easy to sell):** Precision = 0.27, Recall = 0.52, F1-score = 0.35

---

## 11. Confusion Matrix Interpretation
The confusion matrix for our balanced classification model is:
$$\begin{bmatrix} 569 & 576 \\ 195 & 209 \end{bmatrix}$$

* **Interpretation:** 
  * Out of 1,145 actual slow-selling cars (Class 0), the model correctly classified 569.
  * Out of 404 actual fast-selling cars (Class 1), the model correctly predicted 209 (yielding a recall of 52%).
  * **Reason for Low Performance:** The model's overall accuracy of 50.2% is equivalent to random guessing. This is entirely expected because **the target column `days_on_market` (and therefore `is_fast_sale`) is synthetically generated**. Since the target has no real-world correlation or physical relationship with the vehicle's features, a mathematical model cannot identify any patterns, leading to coin-flip predictions.

---

## 12. Priced-Below-Market (Good Deal) Analysis
Using the Polynomial model, we calculated a `price_gap` (Actual Price - Predicted Price). 
* Cars with a negative gap (`price_gap < 0`) are labeled **Good Deal (1)**.
* Cars with a positive gap (`price_gap > 0`) are labeled **Expensive (0)**.

We crossed this with `is_fast_sale` to verify if underpriced cars sell quicker:

| Category | Slow Sale (0) | Fast Sale (1) | Fast Sale Ratio |
| :--- | :---: | :---: | :---: |
| **Priced Above Market (0)** | 2,878 | 926 | **24.3%** |
| **Priced Below Market (1)** | 2,960 | 978 | **24.8%** |

* **Analysis:** Only 24.8% of "Good Deals" sold within 30 days compared to 24.3% of "Expensive" cars. Piyasaya göre ucuz olan araçlar, pahalı olanlara kıyasla belirgin şekilde hızlı satılmamaktadır. This confirms that the synthetic nature of the target variable `days_on_market` creates a random distribution, eliminating any logical link between market discount and sales speed.

---

## 13. General Conclusion
* **Price Prediction:** The vehicle valuation engine built using Polynomial Regression is highly robust, providing reliable market appraisals for used cars.
* **Sellability Analysis:** While the classification accuracy is poor (~50.2%), this is mathematically justified due to the synthetic nature of the sellability metrics. In a real-world scenario, sales velocity is determined by listing quality, page impressions, and pricing strategies rather than synthetic random assignments.
