import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

with open("singapore.pkl","rb") as files:
    model=pickle.load(files)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)
st.set_page_config(
        page_title="Singapore  Resale Flat Prices Predicting",
        layout="wide",
    )
cols = ['month','town','flat_type','block','street_name','storey_range','flat_model','lease_commence_date']
st.title("Singapore  Resale Flat Prices Predicting")

st.header("Prediction of selling price using decision tree Regressor")
st.write("**Please enter the following details to get the predicted selling price**")
mon=st.text_input("Please enter the Month and year in the format YYYY-MM ")
town=st.selectbox("Please select the town",('ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
       'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
       'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
       'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
       'PUNGGOL'))
ft=st.selectbox("Please select the flat type",('1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
       'MULTI GENERATION', 'MULTI-GENERATION'))
block=st.text_input("Please enter the block")
sn=st.text_input("Please enter the street name")
sr=st.selectbox("Please select the storey range",('10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15',
       '19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
       '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10',
       '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
       '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51'))
fas=st.number_input("Please enter the floor area sqm")
lcd=st.selectbox("Please select the lease commence data",(1977, 1976, 1978, 1979, 1984, 1980, 1985, 1981, 1982, 1986, 1972,
       1983, 1973, 1969, 1975, 1971, 1974, 1967, 1970, 1968, 1988, 1987,
       1989, 1990, 1992, 1993, 1994, 1991, 1995, 1996, 1997, 1998, 1999,
       2000, 2001, 1966, 2002, 2006, 2003, 2005, 2004, 2008, 2007, 2009,
       2010, 2012, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022,
       2020))
fm=st.selectbox("Please select the flat model",('IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
       'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE',
       '2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION',
       'PREMIUM APARTMENT', 'Improved', 'New Generation', 'Model A',
       'Standard', 'Apartment', 'Simplified', 'Model A-Maisonette',
       'Maisonette', 'Multi Generation', 'Adjoined flat',
       'Premium Apartment', 'Terrace', 'Improved-Maisonette',
       'Premium Maisonette', '2-room', 'Model A2', 'DBSS', 'Type S1',
       'Type S2', 'Premium Apartment Loft', '3Gen'))
button1=st.button("Predict the Selling Price")

if button1:
    new_sample = {
    'month': mon,
    'town': town,
    'flat_type': ft,
    'block': block,
    'street_name': sn,
    'storey_range': sr,
    'floor_are_sqm': fas,
    'lease_commence_date': lcd,
    'flat_model': fm,
}



new_data_df = pd.DataFrame([new_sample])

# Encode the categorical features using the label encoders
# for col in cols:
#     new_data_df[col] = label_encoders[col].transform(new_data_df[col])
# Scale the new data point using the same scaler
for column in new_data_df.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        le = label_encoders[column]
        new_data_df[column] = le.transform(new_data_df[column])
new_data_scaled = loaded_pipeline.transform(new_data_df)
new_pred = model.predict(new_data_scaled)
# Predict using the trained model
#new_pred = inv_boxcox(new_pred_transformed, boxcox_lambdas['resale_price'])

print("Prediction for the new data point:", new_pred[0])
st.success("The selling is predicted!!")
st.dataframe(new_data_df)
st.dataframe(new_data_scaled)
st.write(f'**Predicted selling price for the new sample: {new_pred[0]}**')