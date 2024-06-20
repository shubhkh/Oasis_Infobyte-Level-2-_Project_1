
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv(r"Housing.csv")

pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

st.title("House Price Prediction! ")

st.number_input("Area of the house")

bedrooms = st.number_input("No. of Berdrooms")

mainroad = st.selectbox("Mainroad",("Yes","No"))

guestroom = st.selectbox("Guestroom",("Yes","No"))

airconditioning = st.selectbox("Airconditioning",("Yes","No"))

parking = st.number_input("No. of Parking")

st.button("Predict Price")


import streamlit as st
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# Function to train the model (from your Jupyter Notebook)
def train_model():
    # Load your dataset
    df = pd.read_csv('Housing.csv')

    # Define feature and target
    X = df[['area', 'bedrooms', 'mainroad', 'guestroom', 'airconditioning', 'parking']]
    y = df['price']

    # Create a column transformer
    column_trans = make_column_transformer(
        (OneHotEncoder(sparse_output=False), ['mainroad', 'guestroom', 'airconditioning']),
        remainder='passthrough'
    )

    # Create a pipeline
    model = make_pipeline(column_trans, LinearRegression())

    # Fit the model
    model.fit(X, y)
    
    return model

# Load the model
model = train_model()

# Streamlit app
st.title('House Price Prediction!')

# Input fields
area = st.number_input('Area of the house', step=0.01)
bedrooms = st.number_input('No. of Bedrooms', step=0.01)
mainroad = st.selectbox('Mainroad', ['yes', 'no'])
guestroom = st.selectbox('Guestroom', ['yes', 'no'])
airconditioning = st.selectbox('Airconditioning', ['yes', 'no'])
parking = st.number_input('No. of Parking', step=0.01)

# Prediction button
if st.button('Predict Price'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'airconditioning': [airconditioning],
        'parking': [parking]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write(f'The predicted price is: INR  {prediction[0]:,.2f}')




