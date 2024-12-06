import streamlit as st
import pandas as pd
import pickle

# Paths to the model and dataset
MODEL_PATH = 'D:/GUVI_Projects/My_Projects/new_horse/Horse/horse_gb_model.pkl'
DATASET_PATH = 'D:/GUVI_Projects/My_Projects/new_horse/Horse/cleaned_final_dataset.csv'

# Load the model dynamically
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the dataset dynamically
def load_dataset():
    try:
        dataset = pd.read_csv(DATASET_PATH)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Dynamically fetch dropdown options for specific features
def get_dropdown_options(dataset, feature):
    if feature in dataset.columns:
        return sorted(dataset[feature].dropna().unique())
    else:
        st.warning(f"Feature '{feature}' not found in the dataset.")
        return []

# Prediction function
def predict(input_data, model, features):
    input_df = pd.DataFrame([input_data], columns=features)
    predictions = model.predict(input_df)
    return predictions[0]

# Footer
def footer():
    st.markdown("""
    ---
    **Creator:** Shubhangi Patil  
    **Project:** Data Science  
    **GitHub Link:** [GitHub Repository](https://github.com/shubhangivspatil)
    """, unsafe_allow_html=True)

# Streamlit App
def main():
    st.title("Horse Race Prediction")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Predict"])

    if page == "Home":
        st.header("Horse Race Prediction Project")
        st.write("""
        ### Project Overview
        This project aims to predict horse racing outcomes using machine learning techniques. The dataset includes detailed information on horse races and individual horses from 1990 to 2020. 
        Given the complexity and inherent unpredictability of horse racing, this project seeks to explore various machine learning models and feature engineering techniques to improve prediction accuracy.
        
        ### Technologies Used
        - **Data Cleansing:** Pandas  
        - **Exploratory Data Analysis (EDA):** Matplotlib, Seaborn, Plotly  
        - **Machine Learning:** Scikit-learn, Imbalanced-learn  
        - **Visualization:** Tableau, Power-BI  
        - **Development Environment:** Jupyter Notebook, Python  

        ### Project Goals
        - **Primary Goal:** To predict the outcome of horse races (`res_win`).  
        - **Secondary Goals:**  
            - To identify significant features affecting race outcomes.  
            - To explore the imbalanced nature of the dataset and develop techniques to handle it.  
            - To create a robust prediction model using historical data.  
        """)

        footer()

    elif page == "Predict":
        st.header("Predict Race Outcomes")

        # Load the dataset for dropdown options
        dataset = load_dataset()
        if dataset is not None:
            # Display dataset overview
            st.subheader("Dataset Overview")
            st.write(dataset.head())
            st.write(f"Dataset contains **{dataset.shape[0]} rows** and **{dataset.shape[1]} columns**.")

            # Prepare features for input
            features = ['trainerName', 'jockeyName', 'course', 'weight', 'distance', 'position', 'RPR', 'TR', 'speed_ratio', 'success_rate']
            user_inputs = {}

            # Collect user inputs dynamically
            for feature in features:
                if feature in dataset.columns:
                    if dataset[feature].dtype == 'object':  # Categorical features
                        options = get_dropdown_options(dataset, feature)
                        user_inputs[feature] = st.selectbox(f"Select value for {feature}:", options)
                    else:  # Numeric features
                        user_inputs[feature] = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.2f")
                else:
                    # Assign default values for missing features
                    if feature == 'speed_ratio':
                        user_inputs[feature] = st.number_input(f"Enter value for {feature} (default = 1.0):", value=1.0, format="%.2f")
                    elif feature == 'success_rate':
                        user_inputs[feature] = st.number_input(f"Enter value for {feature} (default = 0.5):", value=0.5, format="%.2f")

            # Load the model for prediction
            model = load_model()
            if model:
                # Perform prediction
                if st.button("Predict"):
                    try:
                        # Convert user inputs to a single-row DataFrame
                        input_data = pd.DataFrame([user_inputs], columns=features)

                        # Ensure all columns match the model's expected features
                        input_data = input_data[features]

                        # Perform prediction
                        prediction = model.predict(input_data)

                        # Display prediction result
                        st.subheader("Prediction Result")
                        st.write(f"The predicted outcome for this race is: **{'Win' if prediction == 1 else 'No Win'}**")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

        footer()

if __name__ == '__main__':
    main()







