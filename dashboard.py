import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import yfinance as yf
from datetime import datetime
import time
from prophet import Prophet
import joblib
from tensorflow.keras.models import load_model
from pmdarima.arima import auto_arima
from io import BytesIO
import requests
from PIL import Image

# Initialize session state variables
if 'home_page' not in st.session_state:
    st.session_state.home_page = True
if 'selected_ticker' not in st.session_state or st.session_state.selected_ticker not in ['Select a ticker'] + sorted(pd.read_csv('./model_weight.csv')['TICKER'].unique()):
    st.session_state.selected_ticker = 'Select a ticker'

# Set the title of the Streamlit app
st.title('📈 Stock Prediction Dashboard')
st.divider()

# Company ticker lists for each sectors
sectors = {
    "💻 Technologies": ["MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN", "CSCO"],
    "💰 Financial Services": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "BX", "MS", "GS"],
    "🏥 Health Care" : ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN"],
    "🛍️ Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "TJX", "SBUX", "ABNB"],
    "🏭 Industrials": ["GE", "CAT", "UNP", "HON", "UPS", "BA", "RTX", "ADP", "LMT", "DE"],
    "📱 Communication Services": ["GOOG", "META", "NFLX", "DIS", "TMUS", "VZ", "CMCSA", "T", "CHTR"],
    "🏪 Consumer Defensive": ["WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "MNST"],
    "🔋 Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "EPD", "PXD", "OXY"],
    "🔧 Basic materials": [ "LIN", "SHW", "SCCO", "ECL", "CRH", "FCX", "APD", "NUE", "DOW", "CTVA"],
    "🏢 Real Estate": ["PLD", "AMT", "EQIX", "WELL", "PSA", "CCI", "O", "DLR", "CSGP"],
    "⚙️ Utilities": ["NEE", "SO", "DUK", "SRE", "AEP", "PCG", "CEG", "D", "EXC", "XEL"]
}

# Function to display the welcome message and sector stocks
def show_welcome_message():

    st.markdown("""
    ### Da In Kim's 3rd Year Project
    #### Stock Price Prediction Using Ensemble Models
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify;">
    This dashboard presents the outcome of an ensemble model that combines time-series and deep learning models for stock price prediction. 
    <br><br>
    The ensemble model is crafted by integrating Prophet, ARIMA, LSTM, and GRU models, each weighted based on their Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE), and Mean Directional Accuracy (MDA) evaluated during the validation process. 
    <br><br>     
    Models are trained with data up to 2023/11/10
    <br><br>
    The stocks displayed under each sector have been carefully selected based on their market capitalization, representing the top 10 companies within each category according to Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Display sectors and stocks within them
    for sector_name, ticker_list in sectors.items():
        st.markdown(f"##### {sector_name}")
        cols = st.columns(5)
        for i, ticker in enumerate(ticker_list):
            with cols[i % 5]:
                button_style = "text-align: left; width: 100%; margin-bottom: 5px;"
                if st.button(ticker, key=f"btn_{ticker}", help=f"Click to see predictions for {ticker}", kwargs={"style": button_style}):
                    st.session_state.selected_ticker = ticker
                    st.session_state.home_page = False

# Load model weights information
model_weights_df = pd.read_csv('./model_weight.csv')

# Generate a sorted list of tickers
tickers = sorted(model_weights_df['TICKER'].unique())

# Sidebar for ticker selection and home button
home_button = st.sidebar.button('🏠 Go to Home Page')

if home_button:
    st.session_state.home_page = True
    # Reset selected ticker
    st.session_state.selected_ticker = 'Select a ticker'

selected_ticker = st.sidebar.selectbox('Choose a stock ticker:', ['Select a ticker'] + tickers, index=0)

if st.sidebar.button('Submit Ticker'):
    if selected_ticker and selected_ticker != 'Select a ticker':
        st.session_state.selected_ticker = selected_ticker
        st.session_state.home_page = False
    else:
        st.session_state.home_page = True

# Add UoM logo in the sidebar
with st.sidebar:
    st.markdown("""
    <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
    """, unsafe_allow_html=True)
    st.image('uomlogo.png', use_column_width=True)

# Function to show model weights
def show_model_weights(ticker):
    if ticker in model_weights_df['TICKER'].values:
        # Read dataframe to get model weights for selected ticker
        weights = model_weights_df.loc[model_weights_df['TICKER'] == ticker, ['weight_prophet', 'weight_arima', 'weight_lstm', 'weight_gru']].squeeze()
        labels = ['Prophet', 'ARIMA', 'LSTM', 'GRU']
        values = [weights['weight_prophet'], weights['weight_arima'], weights['weight_lstm'], weights['weight_gru']]

        colors = ['#1f77b4',  
          '#2ca02c',  
          '#ff7f0e',  
          '#d62728'] 
        
        # Creating the pie chart
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
        fig.update_traces(textposition='outside', textinfo='percent+label', pull=[0.1, 0, 0, 0], marker=dict(line=dict(color='#000000', width=2)))
        fig.update_layout(title_text=f'Model Weights for {ticker}', title_x=0.5)
        
        # Displaying the pie chart
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"No model weight information available for {ticker}")

# Function to load and show model performance metrics
def show_model_performance(ticker):
    st.write(f"##### Individual Model performance metrics for {ticker}:", unsafe_allow_html=True)
    
    # Define the layout: one column for each model
    col_prophet, col_arima, col_lstm, col_gru = st.columns(4)
    model_names = ['Prophet', 'ARIMA', 'LSTM', 'GRU']
    
    # Iterate over each model and display its metrics
    for model_name, col in zip(model_names, [col_prophet, col_arima, col_lstm, col_gru]):
        file_path = f'./{model_name.lower()}.csv'
        try:
            model_df = pd.read_csv(file_path)
            ticker_metrics = model_df.loc[model_df['TICKER'] == ticker].squeeze()
            
            # Display the metrics within the appropriate column
            with col:
                st.metric(label=f"{model_name} MAPE", value=f"{ticker_metrics['mape']:.2f}%")
                st.metric(label=f"{model_name} MSE", value=f"{ticker_metrics['mse']:.2f}")
                st.metric(label=f"{model_name} MDA", value=f"{ticker_metrics['directional_accuracy']:.2f}%")
        
        except FileNotFoundError:
            st.error(f"The file for {model_name} could not be found. Please check the file path and name.")

# Function to forecast models
def test_models(ticker, start_date, end_date):
    st.text("Testing models...")

    # Function to load models and forecast for period that user set
    prophet(ticker, start_date, end_date)
    arima(ticker, start_date, end_date)
    lstm(ticker, start_date, end_date)
    gru(ticker, start_date, end_date)

    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("Models testing successfully!")

# Function to forecast using prophet
def prophet(ticker, start_date, end_date):
    # Read the CSV file into a DataFrame
    file_path = f'./data/stock_price_data/{ticker}.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Extract 'Close' column and reset the index
    df_prophet = df['Close'].reset_index()

    # Rename the columns to match Prophet's requirements
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    # Create and fit the Prophet model using the training data
    model = Prophet()
    model.fit(df_prophet)
    
    # Make future DataFrame for forecasting
    last_date = df_prophet['ds'].max()
    future_days = (pd.to_datetime(end_date) - last_date).days
    future = model.make_future_dataframe(periods=future_days)
    
    # Forecast using the fitted model
    forecast = model.predict(future)
    
    # Select only 'ds' and 'yhat' for consistency with ARIMA model predictions
    forecast_df = forecast[['ds', 'yhat']]
    
    # Filter predictions to start from start_date
    forecast_df = forecast_df[forecast_df['ds'] >= pd.to_datetime(start_date)]
    
    # Store the forecast in the session state
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = {}
    st.session_state.forecast_data[f'{ticker}_prophet'] = forecast_df
    
    st.write(f"Prophet model training and forecasting for {ticker} completed.")

# Function to forecast using ARIMA
def arima(ticker, start_date, end_date):
    # Read the CSV file into a DataFrame
    file_path = f'./data/stock_price_data/{ticker}.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Extract 'Close' column and reset the index
    df_arima = df['Close'].reset_index()

    # Rename the columns for ARIMA and Convert the 'ds' column to datetime
    df_arima.columns = ['ds', 'y']
    df_arima['ds'] = pd.to_datetime(df_arima['ds'])
    
     # Fit ARIMA model using the training data with auto_arima
    model_auto_arima = auto_arima(df_arima['y'], start_p=1, start_q=1,
                                  test='adf',       
                                  max_p=3, max_q=3, 
                                  m=1,              
                                  d=None,           
                                  seasonal=False,   
                                  start_P=0, 
                                  D=0, 
                                  trace=True,       
                                  error_action='ignore',  
                                  suppress_warnings=True, 
                                  stepwise=True)    

    # Forecast using the fitted ARIMA model
    last_date = df_arima['ds'].max()
    future_days = (pd.to_datetime(end_date) - last_date).days
    forecast, conf_int = model_auto_arima.predict(n_periods=future_days, return_conf_int=True)
    
    # Filter predictions
    last_date = df_arima['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
    
    # Store the forecast in the session state
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = {}
    st.session_state.forecast_data[f'{ticker}_arima'] = forecast_df
    
    st.write(f"ARIMA model training and forecasting for {ticker} completed.")

# Function to create a sequence dataset for time series forecasting
def make_sequence_dataset(feature, label, window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

# Function to forecast using LSTM
def lstm(ticker, start_date, end_date):
    model_path = f'./models/{ticker}_lstm_model.h5'

    # Load stock data using yfinance
    df_ticker = yf.download(ticker, end=end_date)
    df_snp500 = yf.download('^GSPC', end=end_date)
    df_dji = yf.download('^DJI', end=end_date)
    df_nasdaq = yf.download('^IXIC', end=end_date)
    
    df_ticker['Close_snp500'] = df_snp500['Close']
    df_ticker['Close_dji'] = df_dji['Close']
    df_ticker['Close_nasdaq'] = df_nasdaq['Close']

    features = ['Close', 'Volume', 'Close_snp500', 'Close_dji', 'Close_nasdaq']
    feature_df = df_ticker[features]

    # Preprocess: Replace 0 with NaN and drop rows with NaN
    for column in features:
        feature_df[column] = feature_df[column].replace(0, np.nan)

    feature_df = feature_df.dropna()

    # Scale features
    scaled_features = pd.DataFrame(index=feature_df.index)
    for col in features:
        scaler_path = f'./scalers/{col}_scaler.pkl'
        scaler = joblib.load(scaler_path)
        scaled_features[col] = scaler.transform(feature_df[[col]])

    # Convert scaled features to numpy array
    feature_np = scaled_features.to_numpy()
    
    # Create sequences for LSTM input
    X, _ = make_sequence_dataset(feature_np, feature_np[:, 1], window_size=40)

    # Load pre-trained LSTM model
    model = load_model(model_path)
    
    # Make predictions with the model
    predictions = model.predict(X)

    # Inverse scaling for the 'Close' price predictions
    scaler_close = joblib.load('./scalers/Close_scaler.pkl')
    predictions_denormalized = scaler_close.inverse_transform(predictions).flatten()

    future_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    predictions_denormalized = predictions_denormalized[-future_days:]
    
    # Create DataFrame for the forecasted results
    future_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    future_dates = pd.date_range(start=start_date, periods=future_days, freq='D')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': predictions_denormalized})
    
    # Store the forecast in the session state
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = {}
    st.session_state.forecast_data[f'{ticker}_lstm'] = forecast_df
    
    st.write(f"LSTM model training and forecasting for {ticker} completed.")

# Function to forecast using GRU
def gru(ticker, start_date, end_date):
    model_path = f'./models/{ticker}_gru_model.h5'

    # Load stock data using yfinance
    df_ticker = yf.download(ticker, end=end_date)
    df_snp500 = yf.download('^GSPC', end=end_date)
    df_dji = yf.download('^DJI', end=end_date)
    df_nasdaq = yf.download('^IXIC', end=end_date)
    
    df_ticker['Close_snp500'] = df_snp500['Close']
    df_ticker['Close_dji'] = df_dji['Close']
    df_ticker['Close_nasdaq'] = df_nasdaq['Close']

    features = ['Close', 'Volume', 'Close_snp500', 'Close_dji', 'Close_nasdaq']
    feature_df = df_ticker[features]

    # Scale features
    scaled_features = pd.DataFrame(index=feature_df.index)
    for col in features:
        scaler_path = f'./scalers/{col}_scaler.pkl'
        scaler = joblib.load(scaler_path)
        scaled_features[col] = scaler.transform(feature_df[[col]])

    # Convert scaled features to numpy array
    feature_np = scaled_features.to_numpy()
    
    # Create sequences for GRU input
    X, _ = make_sequence_dataset(feature_np, feature_np[:, 1], window_size=30)

    # Load pre-trained GRU model
    model = load_model(model_path)
    
    # Make predictions with the model
    predictions = model.predict(X)

    # Inverse scaling for the 'Close' price predictions
    scaler_close = joblib.load('./scalers/Close_scaler.pkl')
    predictions_denormalized = scaler_close.inverse_transform(predictions).flatten()

    future_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    predictions_denormalized = predictions_denormalized[-future_days:]
    
    # Create DataFrame for the forecasted results
    future_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    future_dates = pd.date_range(start=start_date, periods=future_days, freq='D')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': predictions_denormalized})
    
    # Store the forecast in the session state
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = {}
    st.session_state.forecast_data[f'{ticker}_gru'] = forecast_df
    
    st.write(f"GRU model training and forecasting for {ticker} completed.")

# Function to show prediction of models including ensemble
def show_predictions(ticker, start_date, end_date):
    # Download stock data using yahoo finance api
    df_ticker = yf.download(ticker, end=end_date)

    if not pd.api.types.is_datetime64_any_dtype(df_ticker.index):
        df_ticker.index = pd.to_datetime(df_ticker.index)
    df_ticker.reset_index(inplace=True)

    df_ticker.rename(columns={'index': 'Date'}, inplace=True)
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])

    # Load model weights for ensemble model
    model_weights_df = pd.read_csv('./model_weight.csv')
    weights = model_weights_df.loc[model_weights_df['TICKER'] == ticker, ['weight_prophet', 'weight_arima', 'weight_lstm', 'weight_gru']].squeeze()

    # Display the stock predictoin graph using plotly
    fig = go.Figure()

    # Update the layout of the figure
    fig.update_layout(title_text=f"Stock Price Prediction from {start_date} to {end_date}", xaxis_title='Date', yaxis_title='Price')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the actual data for the selected date range
    df_ticker = df_ticker[(df_ticker['Date'] >= start_date) & (df_ticker['Date'] <= end_date)]
    
    # Plot the actual 'Close' prices for the filtered range
    fig.add_trace(go.Scatter(x=df_ticker['Date'], y=df_ticker['Close'], name='Close', mode='lines'))

    # Define model color for consistency
    model_colors = {
    'prophet': '#1f77b4',
    'arima': '#2ca02c',
    'lstm': '#ff7f0e',
    'gru': '#d62728',
    'ensemble': '#9467bd'
    }

    # Initialize an empty DataFrame for the ensemble predictions
    ensemble_predictions = pd.DataFrame()

    # Calculate weighted predictions for each model and sum them
    for model_key in ['prophet', 'arima', 'lstm', 'gru']:
        full_model_key = f'{ticker}_{model_key}'
        if full_model_key in st.session_state.forecast_data:
            forecast_data = st.session_state.forecast_data[full_model_key]
            forecast_filtered = forecast_data[(forecast_data['ds'] >= start_date) & (forecast_data['ds'] <= end_date)].copy()
            forecast_filtered['weighted_yhat'] = forecast_filtered['yhat'] * weights[f'weight_{model_key}']
            
            if ensemble_predictions.empty:
                ensemble_predictions = forecast_filtered[['ds', 'weighted_yhat']].rename(columns={'weighted_yhat': model_key})
            else:
                ensemble_predictions = ensemble_predictions.merge(forecast_filtered[['ds', 'weighted_yhat']], on='ds', how='outer', suffixes=(False, False)).rename(columns={'weighted_yhat': model_key})

    # Sum the weighted predictions to get the ensemble prediction
    ensemble_predictions['ensemble'] = ensemble_predictions[['prophet', 'arima', 'lstm', 'gru']].sum(axis=1)

    # Post-processing for forecasting
    # Adjusting the starting point of ensemble predictions to match the actual prices
    first_actual_close = df_ticker['Close'].iloc[0]  # First actual closing price in the selected date range
    
    if not ensemble_predictions.empty:
        # First ensemble prediction
        first_ensemble_pred = ensemble_predictions['ensemble'].iloc[0]
        # Calculate the offset
        offset = first_actual_close - first_ensemble_pred
        # Adjust ensemble predictions with the offset
        ensemble_predictions['ensemble'] += offset  
        
        # Plot the adjusted ensemble predictions
        fig.add_trace(go.Scatter(x=ensemble_predictions['ds'], y=ensemble_predictions['ensemble'], name='Ensemble', mode='lines', line=dict(color=model_colors['ensemble'])))

    # Adjusting individual model predictions to match the starting point of actual prices
    for model_name in ['prophet', 'arima', 'lstm', 'gru']:
        model_key = f'{ticker}_{model_name}'
        if model_key in st.session_state.get('forecast_data', {}):
            forecast_data = st.session_state.forecast_data[model_key]
            forecast_filtered = forecast_data[(forecast_data['ds'] >= start_date) & (forecast_data['ds'] <= end_date)].copy()
            
            if not forecast_filtered.empty:
                first_forecast_value = forecast_filtered['yhat'].iloc[0]
                offset = first_actual_close - first_forecast_value  # Calculate offset for each model
                
                forecast_filtered['yhat_adjusted'] = forecast_filtered['yhat'] + offset  # Apply the offset
                
                # Plot the adjusted forecast data
                fig.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat_adjusted'], name=model_name.upper(), mode='lines', line=dict(color=model_colors[model_name]), visible="legendonly"))

    # Display the figure
    st.plotly_chart(fig)

    # Calculate MAPE between ensemble predictions and actual Close prices
    if not ensemble_predictions.empty and not df_ticker.empty:
        df_ticker_for_mape = df_ticker[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'actual'})
        ensemble_for_mape = ensemble_predictions[['ds', 'ensemble']].rename(columns={'ensemble': 'predicted'})
        merged_for_mape = pd.merge(df_ticker_for_mape, ensemble_for_mape, on='ds', how='inner')
        
        # Calculate MAPE
        mape = np.mean(np.abs((merged_for_mape['actual'] - merged_for_mape['predicted']) / merged_for_mape['actual'])) * 100
        mse = np.mean(np.square(merged_for_mape['actual'] - merged_for_mape['predicted']))
        mda = np.mean((np.sign(merged_for_mape['actual'] - merged_for_mape['actual'].shift(1)) == np.sign(merged_for_mape['predicted'] - merged_for_mape['actual'].shift(1)))[1:]) * 100
        
        daily_returns = merged_for_mape['predicted'].pct_change(1)
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        mdd = drawdown.min() * 100

        cumulative_return = (cumulative_returns.iloc[-1] - 1) * 100

        # Disply accuracy indices
        st.write(f"##### Ensemble model performance metrics for {ticker}:", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label=f"MAPE", value=f"{mape:.2f}%")
            st.metric(label=f"Sharpe Ratio", value=f"{sharpe_ratio:.2f}")
            

        with col2:
            st.metric(label=f"MSE", value=f"{mse:.2f}")
            st.metric(label=f"Maximum Drawdown", value=f"{mdd:.2f}%")

        with col3:
            st.metric(label=f"MDA", value=f"{mda:.2f}%")
            st.metric(label=f"Cumulative Returns", value=f"{cumulative_return:.2f}%")
            

# Function to load data and visualize it
def load_data(ticker):
    # Fetch company logo if possible
    logo_url = f"https://eodhd.com/img/logos/US/{ticker}.png"
    try:
        logo_response = requests.get(logo_url)
        if logo_response.status_code == 200:
            logo_image = Image.open(BytesIO(logo_response.content))
            st.image(logo_image, width=100)
    except Exception:
        pass
    
    # Fetch stock information using yahoo finance API
    stock_info = yf.Ticker(str(ticker))
    stock_details = stock_info.info

    string_name = stock_details['longName']
    st.header('**%s**' % string_name)

    # Exander for details of stock from yahoo finance
    with st.expander(f"Stock Details for {ticker}"):
        sector_name = stock_details['sector']
        st.write('**Sector: %s**' % sector_name) 

        # Fetch and display the summary description of stock data
        st.write(f"Summary: {stock_details.get('longBusinessSummary', 'N/A')}")

        # Date range selection
        st.write("**Select Date Range for Historical Stock Data Chart**")
        start_date, end_date = st.columns(2)
        with start_date:
            start_date = st.date_input("Start date", value=pd.to_datetime('2023-11-13'))
        with end_date:
            end_date = st.date_input("End date", value=pd.to_datetime('today'))

        # Fetch and display the historical stock data
        data = stock_info.history(start=start_date, end=end_date)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,open=data['Open'],high=data['High'],low=data['Low'],close=data['Close']))
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(title=f'{ticker} Closing Price', xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig, use_container_width=True)

    if ticker == 'Select a ticker':
        return

    try:
        # Train and test models
        st.subheader("Test Models")

        # Add data range for testing period that user can adjust
        st.write("Select data range for forecasting")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime('2023-11-13'))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime('today'))

        # Button for train models
        if st.button("Test Models"):
            # Forecasting using models
            test_models(ticker, start_date, end_date)

            # Show model prediction
            show_predictions(ticker, start_date, end_date)
            
            # Show model weights
            show_model_weights(ticker)

            # Show model performance metrics
            show_model_performance(ticker)

    except FileNotFoundError:
        st.error('The file could not be found. Please check the ticker symbol.')
        st.table(pd.DataFrame(tickers, columns=['Available Tickers']))

# Show the welcome message on the home page
if st.session_state.home_page or st.session_state.selected_ticker == 'Select a ticker':
    show_welcome_message()

else:
    # If not on the home page, and a ticker is selected or entered, load and display the data
    load_data(st.session_state.selected_ticker)