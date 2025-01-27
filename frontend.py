# import streamlit as st
import requests
# import yfinance as yf
# import pandas as pd
# import time

response =  requests.get("http://127.0.0.1:5000/data")
data = response.json()
#
# # Display data in the Streamlit app
# st.title("Streamlit & Flask Integration!")
# st.write("Backend says: ", data["message"])
# st.write("The value from the backend is: ", data["value"])

# # Streamlit page configuration
# st.set_page_config(page_title="Real-Time Coca-Cola Stock Graph", layout="wide")
#
# # Title of the app
# st.title("Real-Time Coca-Cola Stock Graph")
#
# # Coca-Cola ticker symbol
# TICKER = "IAG"
#
# def get_realtime_data(ticker):
#     """Fetch real-time stock data."""
#     stock_data = yf.download(ticker, period="1y", interval="1h")
#     return stock_data
#
# def plot_stock_graph(stock_data):
#     """Plot the stock graph using Streamlit."""
#     st.line_chart(stock_data["Close"], use_container_width=True)
#
# # Sidebar for refresh control
# refresh_rate = st.sidebar.slider("Refresh rate (seconds):", min_value=1, max_value=60, value=5)
#
# # Main loop to fetch and display data in real-time
# placeholder = st.empty()
#
# while True:
#     try:
#         with placeholder.container():
#             st.subheader(f"Coca-Cola ({TICKER}) Stock Price")
#             stock_data = get_realtime_data(TICKER)
#             if not stock_data.empty:
#                 plot_stock_graph(stock_data)
#             else:
#                 st.warning("No data available. Please check your network connection or ticker symbol.")
#
#         time.sleep(refresh_rate)
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         break
#
#

import streamlit as st
import yfinance as yf
import requests
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch weather data
def get_weather_data(api_key, city="New York"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        return temperature, weather_desc
    else:
        st.error("Failed to fetch weather data!")
        return None, None

# Streamlit App
st.title("Stock and Weather Dashboard")

# User inputs
stock_ticker = st.text_input("Enter Stock Ticker:", "AAPL")
api_key = st.text_input("Enter OpenWeather API Key:")

if api_key and stock_ticker:
    # Fetch stock data
    stock_data = yf.download(stock_ticker, start="2022-01-01", end=dt.datetime.today().strftime("%Y-%m-%d"))

    # Fetch weather data
    temperature, weather_desc = get_weather_data(api_key)

    if stock_data is not None and temperature is not None:
        # Create a subplot with stock data and weather info
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot stock closing prices
        ax1.plot(stock_data.index, stock_data['Close'], label=f"{stock_ticker} Close Price", color="blue")
        ax1.set_ylabel("Stock Price (USD)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_title(f"{stock_ticker} Stock Price and NYC Weather")

        # Plot weather temperature on secondary axis
        ax2 = ax1.twinx()
        ax2.axhline(temperature, color="red", linestyle="--", label=f"NYC Temp: {temperature}°C")
        ax2.set_ylabel("Temperature (°C)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Add weather description as text annotation
        ax1.text(0.5, 0.95, f"NYC Weather: {weather_desc.capitalize()}", transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='center', color="darkgreen")

        # Add legend and show plot
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        st.pyplot(fig)
    else:
        st.error("Error: Unable to fetch stock or weather data!")
else:
    st.warning("Please enter both Stock Ticker and OpenWeather API Key!")
