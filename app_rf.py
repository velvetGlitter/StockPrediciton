import streamlit as st
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta
import joblib
import os

available_tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META']


# Define tickers
training_tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'AMZN']
start_date = '2018-01-01'
end_date = datetime.now()

# Cache data loading function with your preferred approach
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Download and prepare stock data using your preferred approach"""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return pd.DataFrame()

    # Create DataFrame with just Close prices
    data = pd.DataFrame(df['Close'])
    data.columns = [ticker]
    
    # Generate all business days in range
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = data.reindex(all_dates)
    data = data.ffill()
    
    # Calculate returns and technical indicators
    data['Price_1d_Change'] = data[ticker].pct_change(periods=1)
    data['Price_5d_Change'] = data[ticker].pct_change(periods=5)
    data['Price_15d_Change'] = data[ticker].pct_change(periods=15)

    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data[ticker], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(data[ticker], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data[ticker], window=20, window_dev=2)
    data['BB_hband'] = bb.bollinger_hband()
    data['BB_lband'] = bb.bollinger_lband()
    data['BB_mavg'] = bb.bollinger_mavg()
    data['BB_Width'] = (data['BB_hband'] - data['BB_lband']) / data['BB_mavg']
    data['BB_Position'] = (data[ticker] - data['BB_lband']) / (data['BB_hband'] - data['BB_lband'])

    # Target
    data['Target'] = np.where(data[ticker].shift(-1) > data[ticker], 1, 0)
    data['Ticker'] = ticker
    data['Date'] = data.index

    # Clean data
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.ffill().bfill()
    
    return data

@st.cache_data
def load_all_data(tickers, start_date, end_date):
    """Load data for all tickers"""
    all_data = []
    for ticker in tickers:
        df = get_stock_data(ticker, start_date, end_date)
        if not df.empty:
            all_data.append(df)
    return pd.concat(all_data)

# Load data
all_data = load_all_data(training_tickers, start_date, end_date)

# Calculate seasonal patterns
def calculate_seasonal_patterns(data):
    data['DayOfWeek'] = data['Date'].dt.day_name()
    data['Month'] = data['Date'].dt.month_name()
    
    # Calculate returns using the ticker column (now properly named)
    ticker_col = [col for col in data.columns if col in training_tickers][0]
    data['Returns'] = data.groupby('Ticker')[ticker_col].pct_change()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
    
    day_returns = data.groupby('DayOfWeek')['Returns'].mean().reindex(day_order) * 100
    month_returns = data.groupby('Month')['Returns'].mean().reindex(month_order) * 100
    
    pivot_data = data.pivot_table(
        index='DayOfWeek',
        columns='Month',
        values='Returns',
        aggfunc='mean'
    ) * 100
    
    pivot_data = pivot_data.reindex(day_order)
    pivot_data = pivot_data.reindex(columns=month_order)
    
    return day_returns, month_returns, pivot_data

day_returns, month_returns, pivot_data = calculate_seasonal_patterns(all_data)

def prepare_features(df, include_ticker=False):
    # Buat salinan agar data asli tidak terpengaruh
    data = df.copy()

    # Drop kolom yang tidak digunakan
    drop_cols = ['Date', 'Target']
    if not include_ticker:
        drop_cols.append('Ticker')
    
    features = data.drop(columns=drop_cols, errors='ignore')
    
    return features

    # Streamlit App
def main():
    st.title("Stock Market Analysis Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Trends", "Technical Analysis", "Seasonal Patterns", "Prediction"])
    
    with tab1:
        st.subheader("Tech Stocks Price Movement")
        selected_tickers = st.multiselect("Select stocks", training_tickers, default=training_tickers)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for ticker in selected_tickers:
            ticker_data = all_data[all_data['Ticker'] == ticker]
            # Use the ticker name as column for price
            ax.plot(ticker_data['Date'], ticker_data[ticker], label=ticker)
        
        ax.set_title("Price Trends")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Technical Indicators")
        analysis_ticker = st.selectbox("Select stock for analysis", training_tickers)
        ticker_data = all_data[all_data['Ticker'] == analysis_ticker]
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.2, 0.2, 0.1])
        
        # Price and Bollinger Bands
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data[analysis_ticker],
                            name='Close Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['BB_hband'],
                            name='Upper Band', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['BB_lband'],
                            name='Lower Band', line=dict(dash='dash')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['RSI'],
                            name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MACD'],
                            name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['MACD_Signal'],
                            name='Signal'), row=3, col=1)
        
        # Volume would be missing in this approach since we only took Close prices
        # So we'll skip the volume plot in this version
        
        fig.update_layout(height=600, showlegend=True, title_text=f"{analysis_ticker} Technical Indicators")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Seasonal Patterns Analysis")
        
        col1, col2 = st.columns(2)
            
        with col1:
            st.write("### Average Returns by Day of Week")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=day_returns.index, y=day_returns.values, ax=ax)
            plt.ylabel("Average Return (%)")
            st.pyplot(fig)
            
        with col2:
            st.write("### Average Returns by Month")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=month_returns.index, y=month_returns.values, ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel("Average Return (%)")
            st.pyplot(fig)
            
        st.write("### Returns by Day and Month")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_data, cmap='RdYlGn', center=0, annot=True, fmt='.1f', ax=ax)
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Stock Movement Prediction")
        # Ambil data saham menggunakan get_stock_data
    
        # Pilih saham untuk diprediksi
        st.write("Choose the stock tickers to predict.")
        prediction_tickers = st.multiselect("Select stock tickers", available_tickers, default=available_tickers[:3])  # Default ke beberapa ticker pertama

        if not prediction_tickers:
            st.warning("Please select at least one ticker for prediction.")
        else:
            # Input untuk memilih tanggal mulai dan tanggal akhir
            default_end_date = pd.to_datetime('today')  # Tanggal sekarang
            default_start_date = default_end_date - pd.DateOffset(months=3)  # 3 bulan terakhir sebagai default

            prediction_start = st.date_input("Select prediction start date", default_start_date)
            prediction_end = st.date_input("Select prediction end date", default_end_date)
            stock_data = get_stock_data(ticker, str(prediction_start), str(prediction_end)) 
            tickers = stock_data['Ticker'].unique().tolist()
            selected_ticker = st.selectbox("Pilih Ticker:", tickers)
            # Validasi jika prediction_end lebih kecil dari prediction_start
            if prediction_end < prediction_start:
                st.warning("End date must be after start date. Please adjust the dates.")
                prediction_end = prediction_start
            
            all_predictions = pd.DataFrame()

            st.write("Making predictions with saved model for selected stocks:")

            for ticker in prediction_tickers:
                if ticker not in available_tickers:
                    st.warning(f"Ticker {ticker} is not in the list of available tickers. Skipping prediction.")
                    continue  # Skip if ticker is not in available tickers

                st.write(f"\nMaking predictions for {ticker}:")


                if stock_data.empty:
                    st.warning(f"No data available for {ticker} in the specified date range.")
                    continue

                # Temukan nama kolom Close yang benar
                close_col = selected_ticker
                # if not close_cols:
                #     st.error("Tidak ada kolom 'Close' dalam data yang dimuat.")
                #     st.write("Kolom tersedia:", stock_data.columns.tolist())
                #     return
                # close_col = close_cols[0]

                close_prices = stock_data[close_col].copy()

                # Muat model dan scaler
                model_path = 'models/stock_generalized_model.joblib'
                scaler_path = 'models/stock_generalized_model_scaler.joblib'
                features_path = 'models/stock_generalized_model_features.joblib'

                try:
                    model = joblib.load(model_path)
                    # st.write(f"Model loaded from {model_path}")

                    scaler = joblib.load(scaler_path)
                    # st.write(f"Scaler loaded from {scaler_path}")

                    if os.path.exists(features_path):
                        feature_names = joblib.load(features_path)
                        # st.write(f"Feature names loaded from {features_path}")
                    else:
                        feature_names = None
                except Exception as e:
                    st.error(f"Error loading model, scaler, or features: {e}")


                # Siapkan fitur
                features = prepare_features(stock_data, include_ticker=False)

                # Menangani penyelarasan fitur
                if feature_names is not None:
                    # Tambahkan fitur yang hilang
                    for feature in feature_names:
                        if feature not in features.columns:
                            features.loc[:, feature] = 0

                    # Pertahankan hanya fitur yang digunakan saat pelatihan
                    features = features[feature_names]

                # Scale fitur
                features_scaled = scaler.transform(features)

                # Buat prediksi
                predictions = model.predict(features_scaled)
                probabilities = model.predict_proba(features_scaled)

                # Membuat DataFrame hasil
                results = pd.DataFrame({
                    'Close': close_prices,
                    'Actual': stock_data['Target'].map({1: 'UP', 0: 'DOWN'}),
                    'Predicted': np.where(predictions == 1, "UP", "DOWN"),
                    'Probability_Up': probabilities[:, 1],
                    'Probability_Down': probabilities[:, 0]
                }, index=stock_data.index)

                # Tampilkan hasil
                st.write(f"### {ticker} Prediction Results")
                st.write(results.tail())

                # Gabungkan dengan semua prediksi
                all_predictions = pd.concat([all_predictions, results])

            if not all_predictions.empty:
                st.write("### All Predictions")
                st.write(all_predictions)
            else:
                st.warning("No predictions were made.")
            
            st.success("Training, evaluation, and prediction complete!")

def show_rf_model():
    st.title("Random Forest Model")
    main()