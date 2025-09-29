import streamlit as st
from app_LSTM_2 import show_lstm_model
from app_rf import show_rf_model

model_choice = st.radio("Pilih Model :", ("Random Forest Tree", "LSTM"), horizontal=True)

if model_choice == "Random Forest Tree":
    show_rf_model()
elif model_choice == "LSTM":
    show_lstm_model()