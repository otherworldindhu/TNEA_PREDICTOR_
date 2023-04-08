import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('axgb_model.json')

#Caching the model for faster loading
@st.cache

def predict(Year,College_Code, College_Name, Branch_Code,Branch_Name,OC,BC,BCMMBC,SC,SCA,ST):
    
    prediction = model.predict(pd.DataFrame([[Year,College_Code, College_Name, Branch_Code,Branch_Name,OC,BC,BCMMBC,SC,SCA,ST]], columns=['Year','College_Code', 'College_Name', 'Branch_Code','Branch_Name','OC','BC','BCM','MBC','SC','SCA','ST']))
    return prediction


st.title('TNEA Predictor')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.header('Enter the characteristics of the diamond:')