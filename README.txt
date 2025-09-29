use python 3.10.6

# create Virtual Environtment
python3.10.6 -m venv venv

# or
python -m venv venv

# run venv
venv\Scripts\activate

# library
pip install streamlit keras joblib yfinance scikit-learn matplotlib pandas numpy seaborn plotly tensorflow ta

# update yfinance
pip install yfinance --upgrade --no-cache-dir

# run streamlit
streamlit run app.py
