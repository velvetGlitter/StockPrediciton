import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import tensorflow as tf

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import load_model, Sequential
from keras_tuner import BayesianOptimization, Objective
from keras_tuner.engine.hyperparameters import HyperParameters
from keras.layers import LSTM, Bidirectional, Dropout, Dense, BatchNormalization
from keras.optimizers import Adam
import ta  # Technical Analysis library
import joblib

target_tickers = ['AAPL']
start_date = '2017-01-01'
end_date = '2025-03-01'

target_cols = ['Close_AAPL']
window = 10

feature_names = joblib.load('feature_names.save')
feature_scaler = joblib.load('feature_scaler.save')
target_scaler = joblib.load('target_scaler.save')

def get_stock_data(tickers, start_date, end_date):
    """
    Download dan siapin data saham dari ticker yang dimasukin (contoh: AAPL)
    Hasil akhirnya berupa DataFrame dengan nama kolom yang udah dirapihin, misalnya: AAPL_Close, AAPL_High, dll.
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if data.empty:
        print("No data found. Check your tickers and date range.")
        return pd.DataFrame()

    # Ambil data hanya di hari kerja
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = data.reindex(all_dates)

    # Isi data yang kosong ke depan (forward fill), biar ga ada NaN
    data = data.ffill()

    # Flatten MultiIndex (ga kepake, fokus cuma di AAPL)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{ticker}_{field}" for field, ticker in data.columns]
    else:
        data.columns = [f"AAPL_{col}" for col in data.columns]  # fallback, just in case

    print(f"\nDownloaded data shape: {data.shape}")
    return data


def clean_data(df, target_tickers):

    # Ganti nilai inf/-inf jadi NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Track initial data size
    initial_rows = len(df)
    print(f"\nInitial data shape: {df.shape}")

    # 1. Forward-fill hanya raw price/volume kolom untuk AAPL
    for ticker in target_tickers:
        raw_cols = [f'{ticker}_Close', f'{ticker}_High',  # Format: Ticker_Field
                   f'{ticker}_Low', f'{ticker}_Open', f'{ticker}_Volume']
        if set(raw_cols).issubset(df.columns):
            df[raw_cols] = df[raw_cols].ffill()

    # 2. Technical indicators untuk AAPL
    for ticker in target_tickers:
        print(f"\nProcessing {ticker}...")

        # Base columns (format Ticker_Field)
        close_col = f'{ticker}_Close'
        high_col = f'{ticker}_High'
        low_col = f'{ticker}_Low'
        volume_col = f'{ticker}_Volume'

        # Skip jika base columns tidak ada
        if close_col not in df.columns:
            print(f"Missing base columns untuk {ticker}")
            continue

        # Price changes
        for period in [1, 5, 15]:
            df[f'{ticker}_Price_{period}d_Change'] = df[close_col].pct_change(periods=period, fill_method=None)

        # RSI
        df[f'{ticker}_RSI'] = ta.momentum.rsi(df[close_col])

        # MACD
        macd = ta.trend.MACD(df[close_col])
        df[f'{ticker}_MACD'] = macd.macd()
        df[f'{ticker}_MACD_Signal'] = macd.macd_signal()
        df[f'{ticker}_MACD_Hist'] = macd.macd_diff()

        # Volume indicators
        df[f'{ticker}_Volume_Change'] = df[volume_col].pct_change(fill_method=None)
        df[f'{ticker}_OBV'] = ta.volume.OnBalanceVolumeIndicator(df[close_col], df[volume_col]).on_balance_volume()

        # Volatility indicators
        df[f'{ticker}_ATR'] = ta.volatility.AverageTrueRange(df[high_col], df[low_col], df[close_col]).average_true_range()
        df[f'{ticker}_ATR_Pct'] = df[f'{ticker}_ATR'] / df[close_col]

        # Moving averages
        for window in [10, 20, 50]:
            df[f'{ticker}_SMA_{window}'] = ta.trend.SMAIndicator(df[close_col], window=window).sma_indicator()

        df[f'{ticker}_EMA_20'] = ta.trend.ema_indicator(df[close_col], window=20)

        # Price relative to SMA
        df[f'{ticker}_Price_to_SMA20'] = df[close_col] / df[f'{ticker}_SMA_20'] - 1
        df[f'{ticker}_Price_to_SMA50'] = df[close_col] / df[f'{ticker}_SMA_50'] - 1

        # Stochastic Oscillator
        df[f'{ticker}_Stoch_K'] = ta.momentum.stoch(df[high_col], df[low_col], df[close_col])
        df[f'{ticker}_Stoch_D'] = ta.momentum.stoch_signal(df[high_col], df[low_col], df[close_col])

        # Donchian Channel
        df[f'{ticker}_Donchian_High'] = df[high_col].rolling(20).max()
        df[f'{ticker}_Donchian_Low'] = df[low_col].rolling(20).min()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df[close_col])
        df[f'{ticker}_BB_Width'] = bb.bollinger_wband()

        # Feature interaction
        df[f'{ticker}_Volatility_Volume'] = df[f'{ticker}_ATR'] * df[volume_col]
        df[f'{ticker}_RSI_PriceRatio'] = df[f'{ticker}_RSI'] / df[close_col]

        # Bull Market
        df[f'{ticker}_Bull_Market'] = (df[close_col].rolling(50).mean() > df[close_col].rolling(200).mean()).astype(int)

        # Relative Volume
        df[f'{ticker}_RVOL_10'] = df[volume_col] / df[volume_col].rolling(10).mean()

    # 3. lagged features (versi mundur dari fitur utama)
    lagged_features = {}
    lags = [1, 2, 3]
    for ticker in target_tickers:
        for feat in ['Price_1d_Change', 'RSI', 'MACD', 'ATR', 'SMA_20']:
            col = f'{ticker}_{feat}'  # Format: Ticker_Feature
            if col in df.columns:
                for lag in lags:
                    lagged_features[f'{col}_lag{lag}'] = df[col].shift(lag)

    df = pd.concat([df, pd.DataFrame(lagged_features, index=df.index)], axis=1)

   # 4. Final Cleaning bersihin NaN di kolom indikator penting
    tech_indicators = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_50', 'Price_to_SMA50']
    critical_cols = [f"{ticker}_{ind}" for ticker in target_tickers for ind in tech_indicators]
    df = df.dropna(subset=critical_cols)

    # Reporting
    final_rows = len(df)
    print(f"\nCleaning Report:")
    print(f"- Rows removed: {initial_rows - final_rows}")
    print(f"- Final data shape: {df.shape}")
    print(f"- Remaining NaN: {df.isna().sum().sum()}")

    return df

def prepare_lstm_data(data, target_tickers, n_steps=10, test_size=0.2):
    # Make sure kolom target ada
    target_cols = [f'{ticker}_Close' for ticker in target_tickers]
    assert all(col in data.columns for col in target_cols), "Kolom target tidak ditemukan!"

    # Memisahkan fitur dan target
    features = data.drop(columns=target_cols).values
    targets = data[target_cols].values

    # Bangun sequences
    X, y = [], []
    for i in range(n_steps, len(features)):
        X.append(features[i-n_steps:i])
        y.append(targets[i])

    X = np.array(X)
    y = np.array(y)

    # Split data SEBELUM scaling untuk hindari leakage
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scaling hanya menggunakan data train
    feature_scaler = MinMaxScaler().fit(X_train.reshape(-1, X_train.shape[2]))
    target_scaler = RobustScaler().fit(y_train)  # Ganti di sini

    # Transform data
    X_train = np.array([feature_scaler.transform(x) for x in X_train])
    X_test = np.array([feature_scaler.transform(x) for x in X_test])
    y_train = target_scaler.transform(y_train)
    y_test = target_scaler.transform(y_test)

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def lstm_feature_importance(X_train, y_train, feature_names, top_n=20):
    """
    Feature importance untuk data sequence (LSTM) menggunakan Ridge Regression.
    """
    # Flatten data sequence
    n_samples, n_timesteps, n_features = X_train.shape
    X_flat = X_train.reshape((n_samples, n_timesteps * n_features))

    # Train Ridge Regression
    model = Ridge(alpha=1.0)
    model.fit(X_flat, y_train)

    # Hitung importance berdasarkan koefisien absolut
    importance = np.abs(model.coef_)

    # Mapping ke fitur asli dengan rata-rata per fitur
    feature_importance = {}
    for i in range(n_features):
        idx = [j for j in range(i, n_timesteps * n_features, n_features)]
        feature_importance[feature_names[i]] = np.mean(importance[idx])

    # Urutkan dan ambil top_n
    sorted_importance = sorted(feature_importance.items(),
                              key=lambda x: x[1],
                              reverse=True)[:top_n]
    return dict(sorted_importance)

def plot_feature_importance(feature_importance, title="Feature Importance untuk AAPL", top_n=20):
    """
    Plot horizontal bar chart untuk feature importance.
    Mendukung input dict atau list of tuples.
    Return fig agar bisa dipakai di st.pyplot(fig).
    """
    # Pastikan input berupa dict
    if isinstance(feature_importance, list):
        # Jika list of (feature, value)
        if isinstance(feature_importance[0], tuple) and len(feature_importance[0]) == 2:
            feature_importance = dict(feature_importance)
        else:
            raise ValueError("feature_importance harus dict atau list of (feature, value) tuples.")

    # Sort fitur berdasarkan importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)

    # Plot dengan subplots (future-proof untuk Streamlit)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Highest importance on top
    ax.set_xlabel('Importance')
    ax.set_title(title)

    # Tambahkan nilai importance di samping bar
    for i, v in enumerate(importances):
        ax.text(v + max(importances)*0.01, i, f'{v:.4f}', va='center')

    return fig

def build_model(hp):
    model = Sequential()
    units1 = hp.Int('units_layer1', 128, 512, step=64)
    units2 = hp.Int('units_layer2', 64, 256, step=64)
    dropout1 = hp.Float('dropout1', 0.3, 0.6, step=0.1)
    dropout2 = hp.Float('dropout2', 0.2, 0.5, step=0.1)
    dense_units = hp.Int('dense_units', 32, 128, step=32)
    use_bidi = hp.Boolean('use_bidirectional')

    # Input shape HARUS di-hardcode sesuai training (timesteps=10, features=19)
    input_shape = (10, 19)  

    # Layer pertama (BiLSTM)
    if use_bidi:
        model.add(Bidirectional(LSTM(units1, return_sequences=True), input_shape=input_shape))
    else:
        model.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout1))

    # Layer kedua (LSTM)
    model.add(LSTM(units2))
    model.add(Dropout(dropout2))
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(len(target_tickers)))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [1e-4, 5e-4, 1e-3])),
        loss='huber',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model

from keras.models import load_model

def show_lstm_model():
    st.title("📈 LSTM Stock Price Prediction")
    st.markdown("""
    Aplikasi ini memprediksi harga saham menggunakan model LSTM yang sudah di-train. 
    Silakan pilih parameter di sidebar, lalu klik **Run Analysis**.
    """)
    
    with st.sidebar:
        st.header("Parameters")
        ticker = st.text_input("Stock Ticker", "AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime('2017-01-01'))
        end_date = st.date_input("End Date", pd.to_datetime('2025-03-01'))
        show_corr = st.checkbox("Show Correlation Heatmap", value=True)
        run_analysis = st.button("Run Analysis")

    if run_analysis:
        with st.spinner("Processing..."):
            # Load artefak
            feature_names = joblib.load('feature_names.save')
            feature_scaler = joblib.load('feature_scaler.save')
            target_scaler = joblib.load('target_scaler.save')
            window_size = 10  # Atau load dari artefak jika disimpan
            model = load_model('best_model.keras')

            # Data processing (panggil fungsi yang sudah ada)
            raw_data = get_stock_data([ticker], start_date, end_date)
            processed_data = clean_data(raw_data, [ticker])
            features = processed_data[feature_names].values

            # Sequence window
            X_seq = [features[i-window_size:i] for i in range(window_size, len(features))]
            X_seq = np.array(X_seq)

            # Scaling
            X_seq_scaled = np.array([feature_scaler.transform(x) for x in X_seq])
            y_all = processed_data[[f"{ticker}_Close"]].values[window_size:]
            y_scaled = target_scaler.transform(y_all.reshape(-1, 1))

            # Prediction
            y_pred_scaled = model.predict(X_seq_scaled)
            y_pred = target_scaler.inverse_transform(y_pred_scaled)
            y_test_original = target_scaler.inverse_transform(y_scaled)

            # Correlation heatmap
            if show_corr:
                st.subheader("Feature Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(processed_data[feature_names].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)

            # Metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_test_original, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            mask = y_test_original != 0
            mape_score = np.mean(np.abs((y_test_original[mask] - y_pred[mask]) / y_test_original[mask])) * 100
            
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("MAE", f"{mae:.4f}")
            col3.metric("R²", f"{r2:.4f}")
            col4.metric("MAPE", f"{mape_score:.2f}%")

            st.subheader("Price Prediction")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_test_original, label='Actual Prices', color='blue')
            ax.plot(y_pred, label='Predicted Prices', color='orange')
            ax.set_title(f'{ticker} Price Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("Prediction Data")
            results = pd.DataFrame({
                'Date': processed_data.index[-len(y_test_original):],
                'Actual': y_test_original.flatten(),
                'Predicted': y_pred.flatten(),
                'Error': (y_test_original - y_pred).flatten()
            })
            st.dataframe(results.tail(20))

            st.subheader("Feature Importance")
            feature_importance = joblib.load('feature_importance.save')
            try:
                fig = plot_feature_importance(feature_importance, title=f"Feature Importance for {ticker}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal memplot feature importance: {e}")

            # Print shape dan beberapa data awal/akhir
            st.write("Shape y_test_original:", y_test_original.shape)
            st.write("Shape y_pred:", y_pred.shape)

            # Tampilkan 5 data pertama dan terakhir untuk cek urutan
            st.write("y_test_original head:", y_test_original[:5].flatten())
            st.write("y_pred head:", y_pred[:5].flatten())
            st.write("y_test_original tail:", y_test_original[-5:].flatten())
            st.write("y_pred tail:", y_pred[-5:].flatten())

# Panggil fungsi utama
show_lstm_model()
