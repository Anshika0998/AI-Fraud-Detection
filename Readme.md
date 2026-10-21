# AI-Powered Financial Fraud Detection System

An end-to-end machine learning project that identifies potentially
fraudulent credit card transactions using supervised learning.

## Overview

Credit card fraud detection is a challenging classification problem
because fraudulent transactions represent a very small proportion
of all transactions.

This project compares Logistic Regression and Random Forest models
and evaluates them using metrics suitable for imbalanced datasets.

## Features

- Transaction data loading and validation
- Exploratory data analysis
- Missing-value handling
- Stratified training, validation, and test splits
- Logistic Regression and Random Forest comparison
- Validation-based decision threshold selection
- Precision, recall, F1 score, Average Precision, and ROC-AUC
- Saved model for future predictions
- CSV-based transaction prediction
- Interactive Streamlit dashboard
- Downloadable prediction results

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

## Dataset

Dataset: European Credit Card Fraud Detection

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized transaction features, including
Time, Amount, and V1–V28.

The target column is Class:
- 0: Legitimate transaction
- 1: Fraudulent transaction

Download the dataset separately and place `creditcard.csv`
inside the `data/` directory.

## Project Structure

```text
AI-Fraud-Detection/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md