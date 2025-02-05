import streamlit as st
import yfinance as yf
import requests
import datetime as dt
import pandas as pd
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Weather & Stock Analysis Dashboard", layout="wide")

# Function to fetch high-resolution weather forecast data
def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_records = []
        for item in data["list"]:
            dt_object = dt.datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
            weather_records.append({"Datetime": dt_object, "Temperature": item["main"]["temp"]})
        weather_df = pd.DataFrame(weather_records)
        weather_df["Datetime"] = pd.to_datetime(weather_df["Datetime"])
        return weather_df
    else:
        st.error("Failed to fetch weather data! Check API key.")
        return None

# Company ticker mapping
company_ticker_map = {
    "Coca-Cola": "KO",
    "Nestle": "NSRGY",
    "Walmart": "WMT",
    "BP": "BP",
    "Starbucks": "SBUX",
    "Pfizer": "PFE",
    "International Airlines Group": "IAG",
    "TUI Group": "TUI",
    "NVIDIA": "NVDA"
}

# Index ticker mapping with associated country
index_ticker_map = {
    "S&P 500": ("^GSPC", "New York"),
    "FTSE 100": ("^FTSE", "London"),
    "Nikkei 225": ("^N225", "Tokyo")
}

# --- Main Title ---
st.title("Weather & Stock Analysis Dashboard")

tabs = st.tabs(["Company Analysis", "Index Analysis", "Sentiment Analysis", "Combined Analysis"])

with tabs[0]:
    st.header("Company Performance Analysis")
    company_name = st.selectbox("Select Company", options=list(company_ticker_map.keys()), index=0)
    weather_var = st.selectbox("Select Weather Variable", options=["Temperature", "Precipitation"])
    historical_dates = st.date_input("Select Historical Period", [dt.datetime.today() - dt.timedelta(days=30), dt.datetime.today()])
    api_key = "182a5f17141d7682e68478bb12efb8b2"

    if st.button("Run Company Analysis"):
        stock_ticker = company_ticker_map[company_name]
        weather_city = "New York"
        if not api_key or not stock_ticker:
            st.error("Please enter both Stock Ticker and OpenWeather API Key!")
        else:
            start_date = historical_dates[0].strftime("%Y-%m-%d")
            end_date = historical_dates[1].strftime("%Y-%m-%d")
            stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")
            if not stock_data.empty:
                stock_df = stock_data.reset_index()
                stock_df.rename(columns={"Date": "Datetime"}, inplace=True)
                stock_df["Datetime"] = pd.to_datetime(stock_df["Datetime"])
            else:
                stock_df = pd.DataFrame()
            weather_data = get_weather_data(api_key, city=weather_city)
            if not stock_df.empty and weather_data is not None:
                merged_data = pd.merge_asof(stock_df, weather_data, on="Datetime", direction="nearest", tolerance=pd.Timedelta("90min"))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=merged_data["Datetime"], y=merged_data["Close"], mode="lines", name=f"{company_name} Close Price", line=dict(color="blue")))
                if "Temperature" in weather_var:
                    fig.add_trace(go.Scatter(x=merged_data["Datetime"], y=merged_data["Temperature"], mode="lines+markers", name=f"{weather_city} Temperature (°C)", line=dict(color="red"), yaxis="y2"))
                fig.update_layout(title=f"{company_name} Stock Price and {weather_city} Temperature", xaxis_title="Datetime", yaxis=dict(title="Stock Price (USD)", color="blue"), yaxis2=dict(title="Temperature (°C)", color="red", overlaying="y", side="right"), legend=dict(x=0.1, y=1.1), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.header("Index Analysis")
    primary_index = st.selectbox("Select Primary Share Index", options=list(index_ticker_map.keys()), index=0)
    index_dates = st.date_input("Select Index Period", [dt.datetime.today() - dt.timedelta(days=30), dt.datetime.today()])

    if st.button("Run Index Analysis"):
        index_ticker, country = index_ticker_map[primary_index]
        weather_city = country
        start_date = index_dates[0].strftime("%Y-%m-%d")
        end_date = index_dates[1].strftime("%Y-%m-%d")
        index_data = yf.download(index_ticker, start=start_date, end=end_date, interval="1d")
        if not index_data.empty:
            index_df = index_data.reset_index()
            index_df.rename(columns={"Date": "Datetime"}, inplace=True)
            index_df.columns = [col[0] if isinstance(col, tuple) else col for col in index_df.columns]
            index_df["Datetime"] = pd.to_datetime(index_df["Datetime"])
        else:
            index_df = pd.DataFrame()
        weather_data = get_weather_data(api_key, city=weather_city)
        if not index_df.empty and weather_data is not None:
            merged_data = pd.merge_asof(index_df.sort_values("Datetime"), weather_data.sort_values("Datetime"), on="Datetime", direction="nearest", tolerance=pd.Timedelta("90min"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged_data["Datetime"], y=merged_data["Close"], mode="lines", name=f"{primary_index} Close Price", line=dict(color="blue")))
            fig.update_layout(title=f"{primary_index} Performance and {weather_city} Temperature", xaxis_title="Datetime", yaxis=dict(title="Index Value", color="blue"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Sentiment Analysis")
    st.write("Public sentiment analysis results will be displayed here. (Placeholder)")

with tabs[3]:
    st.header("Combined Analysis")
    st.write("Combined analysis of stock, weather, and sentiment data will be shown here. (Placeholder)")
