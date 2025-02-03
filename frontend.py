import streamlit as st
import yfinance as yf
import requests
import datetime as dt
import pandas as pd
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Weather & Stock Analysis Dashboard", layout="wide")


# Function to fetch high-resolution weather forecast data
def get_weather_data(api_key, city="New York"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract timestamps and temperatures
        weather_records = []
        for item in data["list"]:
            dt_object = dt.datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")  # Full timestamp
            weather_records.append({"Datetime": dt_object, "Temperature": item["main"]["temp"]})
        # Convert to DataFrame and ensure single-level Datetime column
        weather_df = pd.DataFrame(weather_records)
        weather_df["Datetime"] = pd.to_datetime(weather_df["Datetime"])
        return weather_df
    else:
        st.error("Failed to fetch weather data! Check API key.")
        return None


# --- Main Title ---
st.title("Weather & Stock Analysis Dashboard")

# --- Sidebar: Data Selection & Options ---
st.sidebar.header("Data Selection & Options")

# Country selection (for demonstration, city is fixed to New York in our weather API call)
country = st.sidebar.selectbox("Select Country", options=["USA", "UK", "Germany", "France", "Japan"], index=0)

# Primary share index selection
primary_index = st.sidebar.selectbox("Select Primary Share Index",
                                     options=["S&P 500", "Dow Jones", "NASDAQ"],
                                     index=0)

# Additional indices (optional)
multi_indices = st.sidebar.multiselect("Select Additional Indices (Optional)",
                                       options=["S&P 500", "Dow Jones", "NASDAQ", "FTSE 100", "DAX"],
                                       default=["S&P 500"])

# Company selection (allowing multiple countries)
companies = st.sidebar.multiselect("Select Companies",
                                   options=["AAPL", "GOOG", "AMZN", "TSLA", "BMW", "SONY"],
                                   default=["AAPL"])

# Weather variable selection (RQ4)
weather_vars = st.sidebar.multiselect("Select Weather Variables",
                                      options=["Temperature", "Humidity", "Wind Speed", "Precipitation"],
                                      default=["Temperature"])

# Historical period selection
historical_dates = st.sidebar.date_input("Select Historical Period",
                                         [dt.datetime.today() - dt.timedelta(days=30), dt.datetime.today()])
if len(historical_dates) != 2:
    st.sidebar.error("Please select a start and an end date for historical analysis.")

# Future prediction period selection (RQ5)
prediction_dates = st.sidebar.date_input("Select Future Prediction Period",
                                         [dt.datetime.today(), dt.datetime.today() + dt.timedelta(days=30)])
if len(prediction_dates) != 2:
    st.sidebar.error("Please select a start and an end date for prediction.")

# Stock ticker input (for simplicity, we use one ticker here)
stock_ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")

# OpenWeather API Key
api_key = st.sidebar.text_input("Enter OpenWeather API Key:")

# Optional: Generative AI summary (RQ6 - Could)
generate_ai = st.sidebar.checkbox("Generate AI Summary Report (Optional)", value=False)
if generate_ai:
    ai_prompt = st.sidebar.text_area("Enter prompt for AI Summary",
                                     "Summarize the recent trends based on the stock and weather data.")
    if st.sidebar.button("Generate AI Report"):
        # Placeholder for Generative AI functionality
        st.sidebar.info("Generating AI report... (This is a placeholder for AI integration)")

# Optional: Public Sentiment Analysis (RQ7)
perform_sentiment = st.sidebar.checkbox("Perform Public Sentiment Analysis", value=True)

# Run Analysis button
if st.sidebar.button("Run Analysis"):
    st.sidebar.info("Analysis started. Results should be ready within 60 seconds.")

    # --- Check Required Inputs ---
    if not api_key or not stock_ticker:
        st.error("Please enter both Stock Ticker and OpenWeather API Key!")
    else:
        # --- Fetch Stock Data ---
        # Use historical period dates selected
        start_date = historical_dates[0].strftime("%Y-%m-%d")
        end_date = historical_dates[1].strftime("%Y-%m-%d")

        # Using daily interval for demonstration; adjust to minute-level if available and within allowed date range
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

        if not stock_data.empty:
            # Reset index to move the date into a column
            stock_df = stock_data.reset_index()
            # yfinance returns the date column as 'Date'. Rename it to 'Datetime' for consistency.
            if 'Date' in stock_df.columns:
                stock_df.rename(columns={"Date": "Datetime"}, inplace=True)
            stock_df["Datetime"] = pd.to_datetime(stock_df["Datetime"])
        else:
            stock_df = pd.DataFrame()

        # --- Fetch Weather Data ---
        # For demonstration, we set city to "New York". In a full system, city could be based on the selected country.
        weather_city = "New York"
        weather_data = get_weather_data(api_key, city=weather_city)

        # --- Validate Data Availability ---
        if stock_df.empty:
            st.error(f"Stock data for {stock_ticker} not available! Check ticker symbol or API restrictions.")
        elif weather_data is None or weather_data.empty:
            st.error("Weather data not available! Check API key and network connection.")
        else:
            # --- Preprocess Data ---
            # Flatten DataFrames if necessary
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.get_level_values(0)
            stock_df["Datetime"] = pd.to_datetime(stock_df["Datetime"])
            if isinstance(weather_data.columns, pd.MultiIndex):
                weather_data.columns = weather_data.columns.get_level_values(0)
            weather_data["Datetime"] = pd.to_datetime(weather_data["Datetime"])

            # Sort by Datetime
            stock_df = stock_df.sort_values("Datetime")
            weather_data = weather_data.sort_values("Datetime")

            # --- Merge Data ---
            merged_data = pd.merge_asof(
                stock_df,
                weather_data,
                on="Datetime",
                direction="nearest",
                tolerance=pd.Timedelta("90min")
            )

            # --- Create Tabs for Analysis ---
            tabs = st.tabs(["Historical Analysis", "Future Prediction", "Sentiment Analysis", "Combined Analysis"])

            with tabs[0]:
                st.header("Historical Performance Analysis")
                # Create interactive Plotly figure for Historical Analysis
                fig = go.Figure()
                # Stock price line
                fig.add_trace(go.Scatter(
                    x=merged_data["Datetime"], y=merged_data["Close"],
                    mode="lines", name=f"{stock_ticker} Close Price",
                    line=dict(color="blue")
                ))
                # Include Temperature if selected as a weather variable
                if "Temperature" in weather_vars:
                    fig.add_trace(go.Scatter(
                        x=merged_data["Datetime"], y=merged_data["Temperature"],
                        mode="lines+markers", name=f"{weather_city} Temperature (°C)",
                        line=dict(color="red"), yaxis="y2"
                    ))
                fig.update_layout(
                    title=f"{stock_ticker} Stock Price and {weather_city} Temperature",
                    xaxis_title="Datetime",
                    yaxis=dict(title="Stock Price (USD)", color="blue"),
                    yaxis2=dict(title="Temperature (°C)", color="red", overlaying="y", side="right"),
                    legend=dict(x=0.1, y=1.1),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                st.header("Future Prediction")
                st.write("Future prediction functionality will be integrated here. (Placeholder)")
                # Future prediction code integrating machine learning (RQ12) can be added here.

            with tabs[2]:
                st.header("Sentiment Analysis")
                if perform_sentiment:
                    st.write("Public sentiment analysis results will be displayed here. (Placeholder)")
                    # Sentiment analysis functionality (RQ7) can be integrated here.
                else:
                    st.write("Sentiment analysis not selected.")

            with tabs[3]:
                st.header("Combined Analysis")
                st.write("Combined analysis of stock, weather, and sentiment data will be shown here. (Placeholder)")

            st.write("""
            ---
            **Note:** This dashboard is designed to be extensible for future indices, weather variables, and time periods.
            It uses open-source libraries for historical data analysis and integrates machine learning models for future prediction.
            """)
