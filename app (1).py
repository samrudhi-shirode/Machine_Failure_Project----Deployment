import streamlit as st
import joblib
import pandas as pd
import pathlib

# Function to load CSS from the 'assets' folder
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load the external CSS
css_path = pathlib.Path("Templates/style.css")
load_css(css_path)
# set the tab title
st.set_page_config("Prediction of Machine Failure")

# Set the page title
st.title("📈Machine Failure Prediction Project 🛠️")

# Set header
st.subheader("By Samrudhi Shirode")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("pre.joblib")
model = joblib.load("model.joblib")


Type = st.selectbox("Type of Machine size",options=['L', 'M','H'])
Air_temperature_K = st.number_input("Air temperature in kelvin")
Process_temperature_K =st.number_input("Process temperature in kelvin")
Rotational_speed_rpm = st.number_input("Rotational_speed_rpm")
Torque_Nm	= st.number_input("Torque_Nm")
Tool_wear_min = st.number_input("Tool_wear_min")
TWF = st.number_input("TWF", min_value=0, max_value=1, step=1)
HDF = st.number_input("HDF", min_value=0, max_value=1, step=1)
PWF = st.number_input("PWF", min_value=0, max_value=1, step=1)
OSF = st.number_input("OSF", min_value=0, max_value=1, step=1)
RNF = st.number_input("RNF", min_value=0, max_value=1, step=1)

# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Predict Machine fail or not", key="green")

if submit:
    data = {
        'Type':[Type],
        'Air_temperature_K':[Air_temperature_K],
        'Process_temperature_K':[Process_temperature_K],
        'Rotational_speed_rpm':[Rotational_speed_rpm],
        'Torque_Nm':[Torque_Nm],
        'Tool_wear_min': [Tool_wear_min],
        'TWF':[TWF],
        'HDF':[HDF],
        'PWF':[PWF],
        'OSF':[OSF],
        'RNF':[RNF]
    }
    # Convert above dictionary into dataframe first
    xnew = pd.DataFrame(data)
    # Apply data cleaning and preprocessing on new data using pre pipeline
    xnew_pre = pre.transform(xnew)
    # predictions
    preds = model.predict(xnew_pre)
    if preds[0]==1:
        op = 'Machine will fail'
        st.subheader(op, key="styledtextfail")
        
    else:
        op = 'Machine will not fail'
        st.subheader(op, key="styledtextnotfail")
       
