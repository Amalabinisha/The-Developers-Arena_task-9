# House Price Prediction Model Evaluation Report

## 1. Project Overview & Objectives
This report details the implementation, methodology, and evaluation of a machine learning workflow designed to predict residential property prices based on key property characteristics.

* **Target Variable:** `Price`
* **Features Used:** `Area`, `Bedrooms`, `Bathrooms`, `Age`, `Location` (City Center, Suburb, Rural), `Property_Type` (House, Villa, Apartment).
* **Excluded Feature:** `Property_ID` (Dropped prior to model training as it serves purely as a unique identifier).

---

## 2. Preprocessing & Technical Architecture

1. **Missing Data Inspection:** Checked the dataset using `.isnull().sum()` to confirm there are zero missing values across all rows and columns.
2. **Categorical Encoding:** Converted non-numeric categorical attributes (`Location`, `Property_Type`) using Scikit-Learn's `OneHotEncoder(drop='first', handle_unknown='ignore')`.
3. **Feature Scaling:** Scaled numerical characteristics (`Area`, `Bedrooms`, `Bathrooms`, `Age`) using `StandardScaler()` to standardize value ranges for gradient descent optimization.
4. **Data Splitting:** Applied an 80/20 train-test split (`X_train`, `X_test`, `y_train`, `y_test`) with a fixed random seed (`random_state=42`) to ensure reproducible evaluation.

---

## 3. Algorithm Development & Modeling Strategy

The modeling strategy covers two distinct implementation layers:

* **Custom Algorithm (From Scratch):** Implemented a gradient-descent-based Linear Regression model (`ScratchLinearRegression`) using vectorized Matrix-Vector multiplication in `numpy` to compute weight updates across epochs.
* **Scikit-Learn Algorithms:** Benchmarked multiple baseline and complex models, including:
  * `LinearRegression`
  * `PolynomialFeatures(degree=2)` with `LinearRegression`
  * `DecisionTreeRegressor(random_state=42)`
  * `RandomForestRegressor(n_estimators=100, random_state=42)`

---

## 4. Performance Metrics & Evaluation Standards

Evaluated models on unseen test data using three core regression metrics:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

* **Primary Evaluation Visualization:** Actual vs. Predicted scatter plot (`predictions_vs_actual.png`) generated to analyze fit along the ideal 1:1 prediction line.
* **Cross-Validation:** Applied k-fold cross-validation (`cv=2`) across models to ensure generalization stability on the training set.

---

## 5. Feature Importance & Key Insights

1. **Dominant Predictor:** Extracting feature importance scores from the Random Forest model confirms that **Property Area (`Area`)** is the strongest individual factor determining property valuation.
2. **Location Premium:** Location indicators (specifically `Location_City Center`) provide the highest secondary coefficient boost to property prices.
3. **Model Recommendation:** Linear Regression models maintain high stability and interpretability on linear pricing trends, while tree ensembles provide effective handling of multi-feature interactions.