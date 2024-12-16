**DS_Placement-Prep-Stage-Project_Horse-Race-Prediction**

# Horse Race Prediction Project

## **Objective**
The primary objective of this project is to predict horse race outcomes (`res_win`: Win or No Win) using advanced machine learning techniques. This project aims to extract meaningful insights from historical data (1990–2020) and develop a robust prediction model capable of handling complex, imbalanced datasets.

---

## **Dataset Overview**
- **Size:** 821,465 records.
- **Timeframe:** 30 years (1990–2020).
- **Features:**
  - **Horse-specific:** `weight`, `RPR` (Race Performance Rating), `TR` (Top speed), `speed_ratio`, `success_rate`.
  - **Race-specific:** `course`, `distance`, `position`, `res_win` (target variable).
  - **Categorical:** `trainerName`, `jockeyName`, `course`.

---

## **Technologies and Tools**
### **Data Preparation**
1. **Pandas:** For data cleaning and merging.
2. **Scikit-learn:** For encoding, preprocessing, and model training.

### **Visualization**
1. **Matplotlib & Seaborn:** Used for static visualizations during EDA.
2. **Plotly:** Created interactive visualizations for deeper exploration.

### **Machine Learning**
1. **Gradient Boosting (final model):** Chosen for its ability to capture complex patterns and handle class imbalances.
2. **Logistic Regression & Random Classifier:** Used as baseline models for performance comparison.

### **Deployment**
- **Streamlit:** Developed a user-friendly interface for real-time predictions.

---

## **Steps Undertaken**

### **1. Data Preprocessing**
1. **Handling Missing Values:**
   - **Numerical Features:** Imputed with the mean.
   - **Categorical Features:** Imputed with the most frequent value.
2. **Feature Encoding:**
   - Categorical features like `trainerName`, `jockeyName`, and `course` were encoded using Label Encoding.
3. **Normalization:**
   - Scaled numerical variables (`weight`, `speed_ratio`) to improve Gradient Boosting performance.
4. **Dataset Preparation:**
   - Merged race-specific and horse-specific datasets for a holistic analysis.

---

### **2. Exploratory Data Analysis (EDA)**
1. **Understanding Distributions:**
   - **Histograms:** Revealed skewness in features like `weight` and `RPR`.
   - **Box Plots:** Highlighted outliers in key metrics like `TR` and `distance`.
2. **Feature Correlations:**
   - Heatmap showed strong positive correlations (`success_rate` ↔ `res_win`, `RPR` ↔ `res_win`).
   - Negative correlation observed between `weight` and performance.
3. **Feature Interactions:**
   - Scatter plots uncovered relationships between features like `speed_ratio` and `position`, providing valuable insights.

---

### **3. Feature Engineering**
1. **Created New Features:**
   - **Speed Ratio:** Indicates the horse's relative pace in prior races.
   - **Success Rate:** Aggregated metric of historical wins for each horse.
2. **Why Feature Engineering?**
   - Introduced domain-specific metrics to improve model understanding.
   - Enhanced representation of complex relationships within the dataset.

---

### **4. Feature Importance Analysis**
1. **Using Mutual Information:**
   - Quantified the dependency between each feature and the target variable (`res_win`).
   - **Top Features Identified:**
     1. `success_rate`
     2. `RPR`
     3. `TR`
     4. `speed_ratio`
     5. `distance`

2. **Why Mutual Information?**
   - Unlike correlation, it detects both linear and non-linear dependencies, making it suitable for identifying key predictors in a complex dataset.

---

### **5. Models Tested**

#### **Random Classifier (Baseline):**
 Best Parameters: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 30, 'bootstrap': False}
- **Confusion Matrix:**
  ```
  [[143147      1]
   [     0  15429]]
  ```
- **Accuracy:** 100% (misleading due to imbalance).

#### **Logistic Regression:**
# Best Parameters for Logistic Regression:
 {'solver': 'liblinear', 'penalty': 'l2', 'C': 10}
- **Confusion Matrix:**
  ```
 [[143147      1]
   [     0  15429]]
  ```
- **Metrics:**
  - Precision (Class 1): 100%.
  - Recall (Class 1): 100%.
  - F1-Score (Class 1): 100%.

#### **Gradient Boosting (Selected Model):**
# Best Parameters for Gradient Boosting:
 {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01}
- **Confusion Matrix:**
  ```
  [[742297      0]
   [     0  79168]]
  ```
- **Metrics:**
  - Precision, Recall, and F1-Score: 100% for both classes.

---

### **6. Why Gradient Boosting?**
1. **Handles Complexity:**
   - Captures subtle, non-linear relationships among features.
2. **Addressed Imbalanced Data:**
   - Outperformed Logistic Regression by effectively learning from minority classes.
3. **Feature Importance:**
   - Provides feature importance rankings, aiding model explainability.
4. **Optimized Performance:**
   - Fine-tuned hyperparameters:
     - `n_estimators`: 100
     - `max_depth`: 3
     - `learning_rate`: 0.01
   - Delivered a balance of precision and recall.
5. **Scalability:**
   - Performs efficiently on large datasets with many features.

---

## **Evaluation and Results**

| **Metric**           | **Gradient Boosting** |
|-----------------------|-----------------------|
| **Accuracy**          | 100%                 |
| **Precision (Class 1)** | 100%               |
| **Recall (Class 1)**  | 100%                 |
| **F1-Score (Class 1)** | 100%               |

---

## **Deployment**
1. **Streamlit Application:**
   - Provides a simple interface for users to input race and horse data.
   - Predicts outcomes (`Win` or `No Win`) in real-time.
   - Displays dataset insights interactively.
2. **Model Integration:**
   - The final Gradient Boosting model is deployed for consistent predictions.

---

## **Key Insights**
1. **Influential Features:**
   - `success_rate` and `RPR` are critical predictors of race outcomes.
2. **Model Selection:**
   - Gradient Boosting outperformed others due to its ability to handle non-linear patterns and imbalances.
3. **High Predictive Power:**
   - Achieved 100% precision, recall, and F1-score for both classes.

---

## **Future Enhancements**
1. **Expand Feature Set:**
   - Incorporate external factors like weather, track conditions, and horse pedigree.
2. **Explore Advanced Models:**
   - Use ensemble methods like XGBoost and LightGBM for additional improvements.
3. **Real-Time Data Integration:**
   - Enable continuous model updates with live race data.

---

## **Deliverables**
1. **Final Model:** Optimized Gradient Boosting model.
2. **Streamlit App:** Interactive tool for predictions and insights.
3. **Cleaned Dataset:** Prepared for additional analysis or model refinement.
