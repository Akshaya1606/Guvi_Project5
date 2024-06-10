
# Singapore  Resale Flat Prices Predicting

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.


## Solution Approach

The solution includes the following steps:

Exploring skewness and outliers in the dataset.

Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.

Model Selection and Training: Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.

Model Evaluation: Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.

Streamlit Web Application: Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.

Deployment on Render: Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.

## Features

- User Friendly
- Dynamic performance
- Colourful Theme
- Valuable Insights from the prediction
## Streamlit App

- Able to give input
- Predict the Reselling Price

## Installation

Install following packages

```bash
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
```
    
## Deployment

To deploy this project run

```bash
  streamlit run copperstream.py
```

The project is deployed in render,please follow the below link.
https://guvi-project5.onrender.com
## Demo

Here is the link of the demo video


https://www.linkedin.com/feed/update/urn:li:activity:7205952478217990146/