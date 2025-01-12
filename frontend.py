import streamlit as st
import requests

response =  requests.get("http://127.0.0.1:5000/data")
data = response.json()

# Display data in the Streamlit app
st.title("Streamlit & Flask Integration")
st.write("Backend says: ", data["message"])
st.write("The value from the backend is: ", data["value"])