import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('axgb_model.json')

#Caching the model for faster loading
@st.cache

def predict(Year,College_Code, College_Name, Branch_Code,Branch_Name,OC,BC,BCM,MBC,SC,SCA,ST):
    
    prediction = model.predict(pd.DataFrame([[Year,College_Code, College_Name, Branch_Code,Branch_Name,OC,BC,BCM,MBC,SC,SCA,ST]], columns=['Year','College_Code', 'College_Name', 'Branch_Code','Branch_Name','OC','BC','BCM','MBC','SC','SCA','ST']))
    return prediction


st.title('TNEA Predictor')
st.image("images\logo.png")
st.header('Enter Your PCM marks and caste:')



marks = st.number_input('PCM marks:', min_value=0, max_value=200, value=1)
caste = st.selectbox('Caste:', ['OC','BC','BCM','MBC','SC','SCA','ST'])
Year = st.selectbox('Year:', ['2017','2018','2019','2020','2021','2022','2023'])

if st.button('Predict College'):
    college = predict(College_Name,OC,BC,BCM,MBC,SC,SCA,ST)
    st.success(f'The predicted College is ${college[0]}')