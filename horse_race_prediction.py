
import os
import streamlit as st
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Paths configuration
log_dir = 'D:/GUVI_Projects/My_Projects/new_horse/Horse'
cleaned_dataset_path = os.path.join(log_dir, 'cleaned_final_dataset_new.csv')  # For user display
mappings_file_path = os.path.join(log_dir, 'mappings.json')  # For backend mapping
cleaned_mapped_dataset_path = os.path.join(log_dir, 'cleaned_dataset_with_mappings.csv')  # Backend operations
gb_model_save_path = os.path.join(log_dir, 'horse_gb_model.pkl')
scaler_save_path = os.path.join(log_dir, 'scaler.pkl')

# Load the model dynamically
def load_model():
    try:
        with open(gb_model_save_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the scaler dynamically
def load_scaler():
    try:
        with open(scaler_save_path, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# Load the dataset dynamically
def load_dataset(dataset_path):
    try:
        dataset = pd.read_csv(dataset_path)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load the mappings for backend conversion
def load_mappings():
    try:
        with open(mappings_file_path, 'r') as file:
            mappings = json.load(file)
        return mappings
    except Exception as e:
        st.error(f"Error loading mappings: {e}")
        return None

# Map user inputs to backend values
def map_user_inputs(user_inputs, mappings):
    backend_inputs = {}
    for key, value in user_inputs.items():
        if key in mappings and value in mappings[key]:
            backend_inputs[key] = mappings[key][value]
        else:
            backend_inputs[key] = value  # Default to user-provided value if no mapping
    return backend_inputs

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
    page = st.sidebar.radio("Go to:", ["Home", "Predict", "Historical Insights"])

    if page == "Home":
        st.header("Horse Race Prediction Project")
        st.write("""
        ### Project Overview
        This project aims to predict horse racing outcomes using machine learning techniques. 
        The dataset includes detailed information on horse races and individual horses from 1990 to 2020. 
        
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

        # Load the datasets and mappings
        display_dataset = load_dataset(cleaned_dataset_path)  # For user-facing dropdowns
        backend_dataset = load_dataset(cleaned_mapped_dataset_path)  # For backend operations
        mappings = load_mappings()

        if display_dataset is not None and backend_dataset is not None and mappings is not None:
            st.write("### Dataset Preview")
            st.dataframe(display_dataset.head(10))

            # Exclude the target column and unwanted columns from features
            features = [col for col in backend_dataset.columns if col not in ['res_win', 'date']]
            user_inputs = {}

            # Collect user inputs dynamically
            for feature in features:
                if feature in display_dataset.columns:
                    if display_dataset[feature].dtype == 'object':  # Categorical features
                        options = sorted(display_dataset[feature].dropna().unique())
                        user_inputs[feature] = st.selectbox(f"Select value for {feature}:", options)
                    else:  # Numeric features
                        user_inputs[feature] = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.2f")

            # Load the model and scaler for prediction
            model = load_model()
            scaler = load_scaler()
            if model and scaler:
                # Perform prediction
                if st.button("Predict"):
                    try:
                        # Map user inputs to backend values
                        backend_inputs = map_user_inputs(user_inputs, mappings)

                        # Convert mapped inputs to a single-row DataFrame
                        input_data = pd.DataFrame([backend_inputs], columns=features)

                        # Scale numeric features
                        numeric_features = [f for f in features if backend_dataset[f].dtype != 'object']
                        input_data[numeric_features] = scaler.transform(input_data[numeric_features])

                        # Perform prediction
                        prediction = model.predict(input_data)

                        # Display prediction result
                        st.subheader("Prediction Result")
                        st.write(f"The predicted outcome for this race is: **{'Win' if prediction == 1 else 'Lose'}**")

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

        footer()

    elif page == "Historical Insights":
        st.header("Historical Data Insights")

        display_dataset = load_dataset(cleaned_dataset_path)
        if display_dataset is not None:
            st.write("### Win Percentage by Trainer")
            trainer_win_rate = display_dataset.groupby('trainerName')['res_win'].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(trainer_win_rate, x='trainerName', y='res_win', title='Win Percentage by Trainer', labels={'res_win': 'Win Percentage', 'trainerName': 'Trainer'}, color='res_win')
            st.plotly_chart(fig)
            st.write("**Insight:** This chart highlights trainers with the highest win percentages. It helps in identifying trainers with consistent performance.")

            st.write("### Horse Weight Distribution")
            fig = px.histogram(display_dataset, x='weight', title='Horse Weight Distribution', labels={'weight': 'Weight (kg)'}, nbins=30, color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig)
            st.write("**Insight:** The weight distribution shows the typical range of horse weights. Understanding this helps in categorizing horses by their physical attributes.")

            st.write("### Performance by Jockey")
            jockey_win_rate = display_dataset.groupby('jockeyName')['res_win'].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(jockey_win_rate, x='jockeyName', y='res_win', title='Top 10 Jockeys by Win Percentage', labels={'res_win': 'Win Percentage', 'jockeyName': 'Jockey'}, color='res_win')
            st.plotly_chart(fig)
            st.write("**Insight:** This chart focuses on the top 10 jockeys, showcasing their win consistency and effectiveness in races.")

            st.write("### Race Distance Distribution")
            fig = px.histogram(display_dataset, x='distance', title='Race Distance Distribution', labels={'distance': 'Distance (meters)'}, nbins=30, color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig)
            st.write("**Insight:** The distance distribution indicates the most common race lengths, which helps in planning race strategies.")

            st.write("### Top Courses by Win Percentage")
            course_win_rate = display_dataset.groupby('course')['res_win'].mean().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(course_win_rate, x='course', y='res_win', title='Top Courses by Win Percentage', labels={'res_win': 'Win Percentage', 'course': 'Course'}, color='res_win')
            st.plotly_chart(fig)
            st.write("**Insight:** This chart identifies the racecourses where horses have performed exceptionally well, helping  focus on strategic locations.")

        footer()

if __name__ == '__main__':
    main()


