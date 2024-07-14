**DS_Placement-Prep-Stage-Project_Horse-Race-Prediction**

**Horse Race Prediction Project**

**Project Overview**
This project aims to predict horse racing outcomes using machine learning techniques. 
The dataset includes detailed information on horse races and individual horses from 1990 to 2020.
Given the complexity and inherent unpredictability of horse racing, this project seeks to explore various machine learning models and feature engineering techniques to improve prediction accuracy.

**Technologies Used**
Data Cleansing: Pandas
Exploratory Data Analysis (EDA): Matplotlib, Seaborn, Plotly
Machine Learning: Scikit-learn, Imbalanced-learn
Visualization: Tableau, Power-BI
Development Environment: Jupyter Notebook, Python

**Project Goals**

Primary Goal: To predict the outcome of horse races (e.g., win or place).
**Secondary Goals:**

To identify significant features affecting race outcomes.
To explore the imbalanced nature of the dataset and develop techniques to handle it.
To create a robust prediction model using historical data.

**Steps Followed**

Data Loading and Merging: Efficiently loaded and merged race and horse data from 1990 to 2020.
Data Preprocessing: Handled missing values, normalized data, and converted categorical variables.
Feature Engineering: Created new features based on past performances and aggregated metrics.
Exploratory Data Analysis (EDA):
Histograms: Displayed the distribution of various features such as weight, position, RPR, and TR.
Correlation Matrix: Analyzed relationships between features to identify significant correlations.
Model Training and Evaluation:
Trained an initial Random Forest model and evaluated its performance.
**Results:**
Accuracy: 90.2%
Precision: 0.91 (for class 0), 0.47 (for class 1)
Recall: 0.99 (for class 0), 0.08 (for class 1)
F1-Score: 0.95 (for class 0), 0.14 (for class 1)
Handling Imbalanced Data: Applied SMOTE to address class imbalance.
Random Forest Model with SMOTE: Achieved improved recall for the minority class.
Results:****
Accuracy: 89.0%
Precision: 0.92 (for class 0), 0.38 (for class 1)
Recall: 0.96 (for class 0), 0.22 (for class 1)
F1-Score: 0.94 (for class 0), 0.28 (for class 1)

Hyperparameter Tuning: Optimized model parameters using Grid Search.
Identified best parameters and retrained the model.
**Best Parameters:** n_estimators: 100, max_depth: 10, min_samples_split: 2, min_samples_leaf: 1
Accuracy: Improved after tuning
Saving Results:
Saved the final cleaned dataset.
Saved the trained Random Forest model with best parameters.

**Key Insights and Conclusions**

Significant features affecting race outcomes were identified.
Prediction accuracy was improved through feature engineering and hyperparameter tuning.
**Future Work**
Includes exploring additional machine learning models and ensemble techniques, and incorporating real-time data for continuous model training and improvement.

