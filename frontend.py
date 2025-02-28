import streamlit as st
import yfinance as yf
import requests
import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from textblob import TextBlob
import nltk
from transformers import pipeline
import scipy.stats as stats

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

# Index ticker mapping with associated country
index_ticker_map = {
    "S&P 500": ("^GSPC", "New York"),
    "FTSE 100": ("^FTSE", "London"),
    "Nikkei 225": ("^N225", "Tokyo")
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
    selected_tab = st.radio("Select Analysis Type",
                            ["Company Analysis", "Index Analysis", "Sentiment Analysis", "Combined Analysis"])

    if selected_tab == "Company Analysis":
        company_name = st.selectbox("Select Company", list(company_ticker_map.keys()), key="company_select")
        weather_city = st.selectbox("Select Weather City", ["New York", "London", "Tokyo"], key="company_weather_city")
        weather_var = st.selectbox("Select Weather Variable", ["Temperature", "Precipitation"],
                                   key="company_weather_var")
        historical_dates = st.date_input("Select Historical Period",
                                         [dt.datetime.today() - dt.timedelta(days=30), dt.datetime.today()],
                                         key="company_hist_dates")
        run_button = st.button("Run Analysis", key="company_run")

    elif selected_tab == "Index Analysis":
        index_name = st.selectbox("Select Index", list(index_ticker_map.keys()), key="index_select")
        weather_city = st.selectbox("Select Weather City", ["New York", "London", "Tokyo"], key="index_weather_city")
        weather_var = st.selectbox("Select Weather Variable", ["Temperature", "Precipitation"], key="index_weather_var")
        historical_dates = st.date_input("Select Historical Period",
                                         [dt.datetime.today() - dt.timedelta(days=30), dt.datetime.today()],
                                         key="index_hist_dates")
        run_button = st.button("Run Analysis", key="index_run")

    elif selected_tab == "Sentiment Analysis":
        run_button = False

    elif selected_tab == "Combined Analysis":
        run_button = False


# Function to preprocess stock data
def preprocess_data(df):
    if df.empty:
        return pd.DataFrame()

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Reset index to convert DatetimeIndex to a column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure the datetime column is named correctly
    if "Date" in df.columns and "Datetime" not in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)

    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Convert to timezone-aware datetime (UTC) and then to GMT
    if df["Datetime"].dt.tz is None:
        df["Datetime"] = df["Datetime"].dt.tz_localize("UTC").dt.tz_convert("GMT")
    else:
        df["Datetime"] = df["Datetime"].dt.tz_convert("GMT")

    return df.sort_values("Datetime")


def get_news_sentiment(stock_name, num_articles=10):
    query = f"{stock_name} stock news"
    news_headlines = []

    # Use Google Search to get news links
    for url in search(query, num_results=num_articles):
        if "news" in url or "finance" in url:
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract the headline (usually the <title> tag)
                title_tag = soup.find("title")
                if title_tag:
                    news_headlines.append(title_tag.text.strip())

            except Exception as e:
                print(f"Error scraping {url}: {e}")

    # Ensure we have news
    if not news_headlines:
        return None

    # Load the classification pipeline with the specified model
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

    total_score = 0
    num_headlines = len(news_headlines)

    # Classify each headline and sum up the scores
    for headline in news_headlines:
        result = pipe(headline)[0]  # Extract the first result
        score = result["score"]  # Extract the score
        total_score += score

        print(f"Headline: {headline}")
        print(f"Sentiment: {result['label']} (Score: {score:.4f})\n")

    # Calculate and print the average score
    if num_headlines > 0:
        avg_score = total_score / num_headlines
        print(f"Average Sentiment Score: {avg_score:.4f}")
    else:
        print("No headlines to process.")

    return avg_score


# Function to create the stock and weather plots
def create_stock_weather_plot(data, title, stock_name=None, index_name=None, weather_city=None, weather_var=None):
    fig = go.Figure()

    # Determine what to label the y-axis based on whether it's a company or index
    if stock_name:
        y_title = "Stock Price (USD)"
        name = f"{stock_name} Hourly Price"
    else:
        y_title = "Index Value"
        name = f"{index_name} Hourly Price"

    fig.add_trace(go.Scatter(x=data["Datetime"], y=data["Close"], mode="lines",
                             name=name, line=dict(color="blue")))

    if weather_var == "Temperature":
        fig.add_trace(go.Scatter(x=data["Datetime"], y=data["Temperature"], mode="lines+markers",
                                 name=f"{weather_city} Temperature (°C)", line=dict(color="red"), yaxis="y2"))
    elif weather_var == "Precipitation":
        fig.add_trace(
            go.Scatter(x=data["Datetime"], y=data["Precipitation"], mode="lines+markers",
                       name=f"{weather_city} Precipitation (mm)", line=dict(color="green"), yaxis="y2"))

    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis=dict(title=y_title, color="blue"),
        yaxis2=dict(title=f"{weather_var} (°C/mm)", color="red", overlaying="y", side="right"),
        hovermode="x unified"
    )

    return fig


# Function to calculate and display correlation analysis
def perform_correlation_analysis(data, price_label, weather_var):
    # Create a copy of the data to avoid modifying the original
    df = data.copy()

    # Drop NA values to ensure valid correlation calculation
    df_clean = df.dropna(subset=["Close", weather_var])

    # Calculate Pearson correlation
    pearson_corr, p_value = stats.pearsonr(df_clean["Close"], df_clean[weather_var])

    # Create a scatter plot to visualize the correlation
    fig = px.scatter(df_clean, x=weather_var, y="Close", trendline="ols",
                     title=f"Correlation between {price_label} and {weather_var}")

    fig.update_layout(
        xaxis_title=f"{weather_var}" + (" (°C)" if weather_var == "Temperature" else " (mm)"),
        yaxis_title=price_label
    )

    # Create a correlation matrix heatmap
    corr_matrix = df_clean[["Close", weather_var]].corr()

    # Create heatmap figure
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="Blues",
        zmin=-1, zmax=1,
        text=[[f"{val:.4f}" for val in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont={"size": 14}
    ))

    heatmap_fig.update_layout(
        title="Correlation Matrix",
        height=400
    )

    # Create a time-windowed correlation figure (rolling correlation)
    window_size = min(24, len(df_clean) // 2)  # Use 24 hours as window size or half the data length if smaller

    # Calculate rolling correlation
    df_clean["Rolling_Correlation"] = df_clean["Close"].rolling(window=window_size).corr(df_clean[weather_var])

    rolling_fig = go.Figure()
    rolling_fig.add_trace(go.Scatter(
        x=df_clean["Datetime"],
        y=df_clean["Rolling_Correlation"],
        mode="lines",
        name=f"{window_size}-hour Rolling Correlation"
    ))

    rolling_fig.update_layout(
        title=f"{window_size}-hour Rolling Correlation Over Time",
        xaxis_title="Datetime",
        yaxis_title="Pearson Correlation Coefficient",
        yaxis=dict(range=[-1, 1])
    )

    return fig, heatmap_fig, rolling_fig, pearson_corr, p_value


# --- Index Analysis ---
if selected_tab == "Index Analysis" and "index_run" in st.session_state and st.session_state.index_run:
    st.header("Index Performance Analysis")

    try:
        index_ticker, associated_city = index_ticker_map[index_name]
        lat, lon = city_coordinates[weather_city]

        # Ensure the date range is two dates (start and end)
        if len(historical_dates) == 1:
            start_date = historical_dates[0]
            end_date = dt.datetime.today()  # Use today as the end date if only one is selected
        else:
            start_date, end_date = historical_dates

        # Convert the dates into the format required for API
        start_date_str, end_date_str = map(lambda d: d.strftime("%Y-%m-%d"), [start_date, end_date])

        # Fetch index data
        with st.spinner('Downloading index data...'):
            index_data = yf.download(index_ticker, start=start_date_str, end=end_date_str, interval="1h")
            index_df = preprocess_data(index_data)

        # Fetch weather data
        with st.spinner('Downloading weather data...'):
            weather_data = get_weather_data(lat, lon, start_date_str, end_date_str)

        if index_df.empty:
            st.error(f"Stock data for {index_ticker} not available! Check ticker symbol or API restrictions.")
        elif weather_data is None or weather_data.empty:
            st.error("Weather data not available! Check API and network connection.")
        else:
            # Final preparation of dataframes
            index_df = index_df.sort_values("Datetime").reset_index(drop=True)
            weather_data = weather_data.sort_values("Datetime").reset_index(drop=True)

            # Merge stock and weather data
            merged_data = pd.merge_asof(
                index_df, weather_data, on="Datetime", direction="nearest", tolerance=pd.Timedelta("1h")
            )

            # Create tabs for Historical and Correlation
            historical_tab, correlation_tab = st.tabs(["Historical Analysis", "Correlation Analysis"])

            with historical_tab:
                st.subheader("Historical Price and Weather Data")
                # Create plot for historical tab
                hist_fig = create_stock_weather_plot(
                    merged_data,
                    f"{index_name} Hourly Price and {weather_city} {weather_var}",
                    index_name=index_name,
                    weather_city=weather_city,
                    weather_var=weather_var
                )
                st.plotly_chart(hist_fig, use_container_width=True)

            with correlation_tab:
                st.subheader("Correlation Analysis")

                # Perform correlation analysis
                scatter_fig, heatmap_fig, rolling_fig, pearson_corr, p_value = perform_correlation_analysis(
                    merged_data,
                    f"{index_name} Value",
                    weather_var
                )

                # Display correlation coefficient and p-value
                st.write(f"### Overall Pearson Correlation: {pearson_corr:.4f}")

                if p_value < 0.05:
                    st.write(f"p-value: {p_value:.4f} (Statistically significant at α = 0.05)")
                else:
                    st.write(f"p-value: {p_value:.4f} (Not statistically significant at α = 0.05)")

                # Interpretation of correlation strength
                if abs(pearson_corr) < 0.1:
                    strength = "Negligible"
                elif abs(pearson_corr) < 0.3:
                    strength = "Weak"
                elif abs(pearson_corr) < 0.5:
                    strength = "Moderate"
                elif abs(pearson_corr) < 0.7:
                    strength = "Strong"
                else:
                    strength = "Very strong"

                direction = "positive" if pearson_corr > 0 else "negative"

                st.write(
                    f"Interpretation: {strength} {direction} correlation between {index_name} and {weather_city} {weather_var}")

                # Display scatter plot
                st.plotly_chart(scatter_fig, use_container_width=True)

                # Display heatmap
                st.plotly_chart(heatmap_fig, use_container_width=True)

                # Display rolling correlation
                st.plotly_chart(rolling_fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# --- Company Analysis ---
if selected_tab == "Company Analysis" and "company_run" in st.session_state and st.session_state.company_run:
    st.header("Company Performance Analysis")

    try:
        stock_ticker = company_ticker_map[company_name]
        lat, lon = city_coordinates[weather_city]

        # Ensure the date range is two dates (start and end)
        if len(historical_dates) == 1:
            start_date = historical_dates[0]
            end_date = dt.datetime.today()  # Use today as the end date if only one is selected
        else:
            start_date, end_date = historical_dates

        # Convert the dates into the format required for API
        start_date_str, end_date_str = map(lambda d: d.strftime("%Y-%m-%d"), [start_date, end_date])

        # Fetch stock data
        with st.spinner('Downloading stock data...'):
            stock_data = yf.download(stock_ticker, start=start_date_str, end=end_date_str, interval="1h")
            stock_df = preprocess_data(stock_data)

        # Fetch weather data
        with st.spinner('Downloading weather data...'):
            weather_data = get_weather_data(lat, lon, start_date_str, end_date_str)

        if stock_df.empty:
            st.error(f"Stock data for {stock_ticker} not available! Check ticker symbol or API restrictions.")
        elif weather_data is None or weather_data.empty:
            st.error("Weather data not available! Check API and network connection.")
        else:
            # Final preparation of dataframes
            stock_df = stock_df.sort_values("Datetime").reset_index(drop=True)
            weather_data = weather_data.sort_values("Datetime").reset_index(drop=True)

            # Merge stock and weather data
            merged_data = pd.merge_asof(
                stock_df, weather_data, on="Datetime", direction="nearest", tolerance=pd.Timedelta("1h")
            )

            # Create tabs for Historical and Correlation
            historical_tab, correlation_tab = st.tabs(["Historical Analysis", "Correlation Analysis"])

            with historical_tab:
                st.subheader("Historical Price and Weather Data")
                # Create plot for historical tab
                hist_fig = create_stock_weather_plot(
                    merged_data,
                    f"{company_name} Hourly Price and {weather_city} {weather_var}",
                    stock_name=company_name,
                    weather_city=weather_city,
                    weather_var=weather_var
                )
                st.plotly_chart(hist_fig, use_container_width=True)

            with correlation_tab:
                st.subheader("Correlation Analysis")

                # Perform correlation analysis
                scatter_fig, heatmap_fig, rolling_fig, pearson_corr, p_value = perform_correlation_analysis(
                    merged_data,
                    f"{company_name} Stock Price",
                    weather_var
                )

                # Display correlation coefficient and p-value
                st.write(f"### Overall Pearson Correlation: {pearson_corr:.4f}")

                if p_value < 0.05:
                    st.write(f"p-value: {p_value:.4f} (Statistically significant at α = 0.05)")
                else:
                    st.write(f"p-value: {p_value:.4f} (Not statistically significant at α = 0.05)")

                # Interpretation of correlation strength
                if abs(pearson_corr) < 0.1:
                    strength = "Negligible"
                elif abs(pearson_corr) < 0.3:
                    strength = "Weak"
                elif abs(pearson_corr) < 0.5:
                    strength = "Moderate"
                elif abs(pearson_corr) < 0.7:
                    strength = "Strong"
                else:
                    strength = "Very strong"

                direction = "positive" if pearson_corr > 0 else "negative"

                st.write(
                    f"Interpretation: {strength} {direction} correlation between {company_name} and {weather_city} {weather_var}")

                # Display scatter plot
                st.plotly_chart(scatter_fig, use_container_width=True)

                # Display heatmap
                st.plotly_chart(heatmap_fig, use_container_width=True)

                # Display rolling correlation
                st.plotly_chart(rolling_fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# --- Sentiment Analysis ---
if selected_tab == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    selected_stock = st.selectbox("Select Stock or Index",
                                  list(company_ticker_map.keys()) + list([k for k in index_ticker_map.keys()]))

    if st.button("Analyse Sentiment", key="sentiment_run"):
        with st.spinner("Analyzing sentiment..."):
            sentiment_score = get_news_sentiment(selected_stock)
            if sentiment_score is not None:
                st.write(f"Sentiment score for {selected_stock}: {sentiment_score:.4f}")
            else:
                st.warning("No news headlines found for this stock/index.")

# --- Combined Analysis ---
if selected_tab == "Combined Analysis":
    st.header("Combined Analysis")
    st.write("Combined analysis of stock, weather, and sentiment data will be shown here. (Placeholder)")