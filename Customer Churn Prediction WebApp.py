# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:08:41 2025

@author: Divya
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the trained model pipeline
with open('trained_pipeline.sav', 'rb') as file:
    loaded_model = pickle.load(file)

def customer_churn_prediction(new_data_df):
    """Predicts customer churn based on input data."""
    prediction = loaded_model.predict(new_data_df) 
    return "Exited" if prediction[0] == 1 else "Did Not Exit"

def main():
    # Streamlit Web App Title
    st.title("Customer Churn Prediction Web App")

    # User Input Fields
    CreditScore = st.text_input("Credit score: ")
    Geography = st.text_input("Country: ")  
    Gender = st.text_input("Gender: ")  
    Age = st.text_input("Age: ")
    Tenure = st.text_input("Tenure: ")
    Balance = st.text_input("Account Balance: ")
    NumOfProducts = st.text_input("Number of products: ")
    HasCrCard = st.radio("Has credit card?", [1, 0])
    IsActiveMember = st.radio("Is active member?", [1, 0])
    EstimatedSalary = st.text_input("Salary: ")

    # Store inputs in a DataFrame
    new_data_df = pd.DataFrame([{
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }])

    # Prediction Button
    if st.button("Customer Churn Prediction Result"):
        result = customer_churn_prediction(new_data_df)
        st.success(result)

if __name__ == '__main__':
    main()
