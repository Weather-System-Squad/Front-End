import streamlit as st
import yfinance as yf
import requests
import datetime as dt
import pandas as pd
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Weather & Stock Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")


# Function to fetch historical weather data from Open-Meteo
def get_weather_data(lat, lon, start_date, end_date):
    start_date = start_date.strftime("%Y-%m-%d") if isinstance(start_date, dt.date) else start_date
    end_date = end_date.strftime("%Y-%m-%d") if isinstance(end_date, dt.date) else end_date

    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation&timezone=Europe/London"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather_records = []

        timestamps = data["hourly"]["time"]
        temperatures = data["hourly"]["temperature_2m"]
        precipitation = data["hourly"].get("precipitation", [None] * len(timestamps))

        for time, temp, precip in zip(timestamps, temperatures, precipitation):
            weather_records.append({"Datetime": pd.to_datetime(time), "Temperature": temp, "Precipitation": precip})

        weather_df = pd.DataFrame(weather_records)

        # Convert to timezone-aware datetime (Europe/London) and then to GMT
        if weather_df["Datetime"].dt.tz is None:
            weather_df["Datetime"] = weather_df["Datetime"].dt.tz_localize("Europe/London").dt.tz_convert("GMT")
        else:
            weather_df["Datetime"] = weather_df["Datetime"].dt.tz_convert("GMT")

        return weather_df

    else:
        st.error("Failed to fetch weather data! Check API or parameters.")
        return None


# Company ticker mapping
company_ticker_map = {
    "Coca-Cola": "KO", "Nestle": "NSRGY", "Walmart": "WMT", "BP": "BP",
    "Starbucks": "SBUX", "Pfizer": "PFE", "International Airlines Group": "IAG",
    "TUI Group": "TUI", "NVIDIA": "NVDA"
}

# City coordinates mapping
city_coordinates = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6895, 139.6917),
}

# Sidebar for navigation and inputs
with st.sidebar:
    st.header("Navigation")
    selected_tab = st.radio("Select Tab", ["Company Analysis", "Sentiment Analysis", "Combined Analysis"])
    company_name = st.selectbox("Select Company", options=list(company_ticker_map.keys()))
    weather_city = st.selectbox("Select Weather City", options=list(city_coordinates.keys()))
    weather_var = st.selectbox("Select Weather Variable", options=["Temperature", "Precipitation"])
    historical_dates = st.date_input("Select Historical Period",
                                     [dt.datetime.today() - dt.timedelta(days=30), dt.datetime.today()])
    run_button = st.button("Run Analysis")


# Function to preprocess stock data
def preprocess_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Datetime" not in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Convert to timezone-aware datetime (UTC) and then to GMT
    if df["Datetime"].dt.tz is None:
        df["Datetime"] = df["Datetime"].dt.tz_localize("UTC").dt.tz_convert("GMT")
    else:
        df["Datetime"] = df["Datetime"].dt.tz_convert("GMT")

    return df.sort_values("Datetime")


# --- Company Analysis ---
if selected_tab == "Company Analysis" and run_button:
    st.header("Company Performance Analysis")
    stock_ticker = company_ticker_map[company_name]
    lat, lon = city_coordinates[weather_city]

    # Ensure the date range is two dates (start and end)
    if len(historical_dates) == 1:
        start_date = historical_dates[0]
        end_date = dt.datetime.today()  # Use today as the end date if only one is selected
    else:
        start_date, end_date = historical_dates

    # Convert the dates into the format required for API
    start_date, end_date = map(lambda d: d.strftime("%Y-%m-%d"), [start_date, end_date])

    # Fetch stock data
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1h")
    stock_df = preprocess_data(stock_data.reset_index()) if not stock_data.empty else pd.DataFrame()

    # Fetch weather data
    weather_data = get_weather_data(lat, lon, start_date, end_date)

    if stock_df.empty:
        st.error(f"Stock data for {stock_ticker} not available! Check ticker symbol or API restrictions.")
    elif weather_data is None or weather_data.empty:
        st.error("Weather data not available! Check API and network connection.")
    else:
        stock_df = stock_df.sort_values("Datetime").reset_index(drop=True)
        weather_data = weather_data.sort_values("Datetime").reset_index(drop=True)

        # Merge stock and weather data
        merged_data = pd.merge_asof(
            stock_df, weather_data, on="Datetime", direction="nearest", tolerance=pd.Timedelta("1h")
        )

        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged_data["Datetime"], y=merged_data["Close"], mode="lines",
                                 name=f"{company_name} Hourly Price", line=dict(color="blue")))

        if weather_var == "Temperature":
            fig.add_trace(go.Scatter(x=merged_data["Datetime"], y=merged_data["Temperature"], mode="lines+markers",
                                     name=f"{weather_city} Temperature (°C)", line=dict(color="red"), yaxis="y2"))
        elif weather_var == "Precipitation":
            fig.add_trace(
                go.Scatter(x=merged_data["Datetime"], y=merged_data["Precipitation"], mode="lines+markers",
                           name=f"{weather_city} Precipitation (mm)", line=dict(color="green"), yaxis="y2"))

        fig.update_layout(
            title=f"{company_name} Hourly Price and {weather_city} {weather_var}",
            xaxis_title="Datetime",
            yaxis=dict(title="Stock Price (USD)", color="blue"),
            yaxis2=dict(title=f"{weather_var} (°C/mm)", color="red", overlaying="y", side="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Sentiment Analysis ---
if selected_tab == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    st.write("Public sentiment analysis results will be displayed here. (Placeholder)")

# --- Combined Analysis ---
if selected_tab == "Combined Analysis":
    st.header("Combined Analysis")
    st.write("Combined analysis of stock, weather, and sentiment data will be shown here. (Placeholder)")
