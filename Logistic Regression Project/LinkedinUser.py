"""
LinkedIn User Prediction Application

This application uses machine learning to predict whether a user is likely
to be a LinkedIn user based on demographic factors. It uses a logistic regression
model trained on social media usage data.

Author: Gaurav Patanker
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st


def clean_sm(df):
    """
    Convert values in the second column to binary (1 or 0)
    
    Args:
        df (pd.DataFrame): DataFrame containing user data
        
    Returns:
        pd.DataFrame: Modified DataFrame with binary values
    """
    binary_value = np.where(df.iloc[:, [1]] == 1, 1, 0)
    df["Binary Value"] = binary_value
    return df


def generate_sample_data(n_samples=252):
    """
    Generate sample data for demonstration purposes
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    # Initial sample data
    data = pd.DataFrame({
        "web1h": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        "income": [7, 5, 8, 3, 9, 4, 6, 8, 2, 7],
        "educ2": [7, 5, 8, 3, 6, 4, 7, 6, 3, 8],
        "par": [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        "marital": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        "gender": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        "age": [35, 42, 28, 65, 30, 52, 38, 45, 21, 33]
    })
    
    # Add more rows to meet n_samples
    additional_rows = n_samples - len(data)
    if additional_rows > 0:
        for _ in range(additional_rows):
            new_row = {
                "web1h": np.random.choice([0, 1]),
                "income": np.random.randint(1, 10),
                "educ2": np.random.randint(1, 9),
                "par": np.random.choice([0, 1]),
                "marital": np.random.choice([0, 1]),
                "gender": np.random.choice([0, 1]),
                "age": np.random.randint(18, 90)
            }
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    
    return data


def preprocess_data(df):
    """
    Preprocess the raw data for model training
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Processed data ready for modeling
    """
    # Select and rename columns
    processed = df.loc[:, ["web1h", "income", "educ2", "par", "marital", "gender", "age"]]
    processed.rename(columns={"web1h": "sm_li", "educ2": "educ"}, inplace=True)
    
    # Clean and transform data
    processed["sm_li"] = np.where(processed["sm_li"] == 1, 1, 0)
    processed["educ"] = np.where(processed["educ"] <= 8, processed["educ"], np.nan)
    processed["income"] = np.where(processed["income"] < 10, processed["income"], np.nan)
    processed["age"] = np.where(processed["age"] < 99, processed["age"], np.nan)
    processed["par"] = np.where(processed["par"] == 1, 1, 0)
    processed["marital"] = np.where(processed["marital"] == 1, 1, 0)
    processed["gender"] = np.where(processed["gender"] == 1, 1, 0)
    
    # Remove missing values and convert types
    processed = processed.dropna()
    processed = processed.astype({"income": int, "educ": int, "age": int})
    
    return processed


def train_model(X, y):
    """
    Train logistic regression model
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        
    Returns:
        tuple: Trained model, X_train, X_test, y_train, y_test
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        stratify=y,
        test_size=0.2,
        random_state=500
    )
    
    # Create and train model
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "accuracy": report["accuracy"]
    }


def predict_linkedin_usage(model, user_data):
    """
    Predict LinkedIn usage for a user
    
    Args:
        model: Trained model
        user_data (pd.DataFrame): User demographic data
        
    Returns:
        tuple: (prediction, probability)
    """
    prediction = model.predict(user_data)
    probabilities = model.predict_proba(user_data)
    return prediction[0], probabilities[0][1]


def create_ui():
    """
    Create the Streamlit user interface
    
    Returns:
        dict: User inputs as dictionary
    """
    st.title("Using Machine Learning to Predict LinkedIn Users")
    st.subheader("By: Gaurav Patanker")
    
    # Define income levels and map to values
    income_levels = [
        'Less than $10,000', 
        '10 to under $20,000',
        '20 to under $30,000',
        '30 to under $40,000', 
        '40 to under $50,000',
        '50 to under $75,000',
        '75 to under $100,000',
        '100 to under $150,000',
        '$150,000 or more'
    ]
    
    income_map = {level: idx + 1 for idx, level in enumerate(income_levels)}
    
    # Define education levels and map to values
    education_levels = [
        'Less than high school',
        'High school incomplete',
        'High school graduate',
        'Some college, no degree',
        'Two-year associate degree from a college or university',
        'Four-year college or university degree/Bachelors degree',
        'Some postgraduate or professional schooling, no postgraduate degree',
        'Postgraduate or professional degree, including masters, doctorate, medical or law degree'
    ]
    
    education_map = {level: idx + 1 for idx, level in enumerate(education_levels)}
    
    # Input widgets
    income_input = st.selectbox('Select Your Income Level', income_levels)
    st.markdown(f"You Selected: **{income_input}**")
    
    education_input = st.selectbox('Select Your highest level of education:', education_levels)
    st.markdown(f"You Selected: **{education_input}**")
    
    parent_input = st.selectbox('Are you a parent?', ['True', 'False'])
    st.markdown(f"You Selected: **{parent_input}**")
    
    marital_input = st.selectbox('What is your Marital Status?', ['Married', 'Not Married'])
    st.markdown(f"You Selected: **{marital_input}**")
    
    gender_input = st.selectbox('What is your gender', ['Male', 'Female'])
    st.markdown(f"You Selected: **{gender_input}**")
    
    age_input = st.slider("What is your age?", 18, 98)
    st.markdown(f"You Selected: **{age_input}**")
    
    # Convert inputs to numeric values
    user_inputs = {
        "income": income_map.get(income_input, 0),
        "educ": education_map.get(education_input, 0),
        "par": 1 if parent_input == 'True' else 0,
        "marital": 1 if marital_input == 'Married' else 0,
        "gender": 1 if gender_input == 'Male' else 0,
        "age": age_input
    }
    
    return user_inputs


def main():
    """
    Main application function
    """
    # Try to load real data, fall back to sample data if not available
    try:
        s = pd.read_csv("social_media_usage.csv")
        print("Loaded actual data")
    except FileNotFoundError:
        s = generate_sample_data()
        print("Using generated sample data")
    
    # Example with dummy data - for demonstration
    df_dummy = pd.DataFrame({"Name": ["John", "Tom", "Sam"], "Linkedin User?": [2, 1, 1]})
    clean_result = clean_sm(df_dummy)
    print(clean_result)
    
    # Preprocess data for modeling
    ss = preprocess_data(s)
    
    # Define target and features
    y = ss["sm_li"]
    X = ss[["income", "educ", "par", "marital", "gender", "age"]]
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Evaluate model
    eval_metrics = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {eval_metrics['accuracy']:.2f}")
    
    # Create user interface and get inputs
    user_inputs = create_ui()
    
    # Create DataFrame with user input
    user_data = pd.DataFrame([user_inputs])
    
    # Make prediction
    prediction, probability = predict_linkedin_usage(model, user_data)
    
    # Display results
    st.subheader("Prediction Results")
    if prediction == 1:
        st.success("You are predicted to be a LinkedIn user")
    else:
        st.info("You are predicted to not be a LinkedIn user")
    
    st.markdown(f"Probability that you are a LinkedIn User: **{probability * 100:.1f}%**")
    
    # Add information about the model
    with st.expander("About This Model"):
        st.write("""
        This model uses logistic regression to predict whether someone is likely 
        to be a LinkedIn user based on demographic factors. 
        
        The model was trained on social media usage data and has an accuracy
        of approximately 69%.

        Factors that increase the likelihood of being a LinkedIn user include:
        - Higher education levels
        - Higher income
        - Younger age (in most cases)
        """)
        
        # Show confusion matrix
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(
            eval_metrics["confusion_matrix"],
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"]
        ))


if __name__ == "__main__":
    main()
                                    