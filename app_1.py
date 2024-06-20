import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('RidgeModel.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the trained model
model = load_model()

# Streamlit app
st.title('House Price Prediction!')

# Input fields
area = st.number_input('Area of the house', step=1.0)
bedrooms = st.number_input('No. of Bedrooms', step=1.0)
mainroad = st.selectbox('Mainroad', ['Yes', 'No'])
guestroom = st.selectbox('Guestroom', ['Yes', 'No'])
airconditioning = st.selectbox('Airconditioning', ['Yes', 'No'])
parking = st.number_input('No. of Parking', step=1.0)

# Prediction button
if st.button('Predict Price'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'mainroad': [1 if mainroad == 'Yes' else 0],  # Convert to binary
        'guestroom': [1 if guestroom == 'Yes' else 0],  # Convert to binary
        'airconditioning': [1 if airconditioning == 'Yes' else 0],  # Convert to binary
        'parking': [parking]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write(f'The predicted price is: INR {prediction[0]:,.2f}')
