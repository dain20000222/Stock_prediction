import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os
import numpy as np

# Initialize session state variables
if 'home_page' not in st.session_state:
    st.session_state.home_page = True
if 'selected_ticker' not in st.session_state or st.session_state.selected_ticker not in ['Select a ticker'] + sorted(pd.read_csv('./model_weight.csv')['TICKER'].unique()):
    st.session_state.selected_ticker = 'Select a ticker'

# Set the title of the Streamlit app
st.title('📈 Stock Price Prediction Dashboard')
st.markdown("---")

sectors = {
    "💻 Technologies": ["MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN", "CSCO"],
    "💰 Financial Services": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "BX", "MS", "GS"],
    "🏥 Health Care" : ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN"],
    "🛍️ Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NIKE", "LOW", "BKNG", "TJX", "SBUX", "ABNB"],
    "🏭 Industrials": ["GE", "CAT", "UNP", "HON", "UPS", "BA", "RTX", "ETN", "ADP", "LMT"],
    "📱 Communication Services": ["GOOG", "META", "NFLX", "DIS", "TMUS", "VZ", "CMCSA", "T", "SPOT", "DASH"],
    "🏪 Consumer Defensive": ["WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT"],
    "🔋 Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "EPD", "PXD", "OXY"],
    "🔧 Basic materials": [ "LIN", "SHW", "SCCO", "ECL", "CRH", "FCX", "APD", "NUE", "DOW"],
    "🏢 Real Estate": ["PLD", "AMT", "EQIX", "SPG", "WELL", "PSA", "CCI", "O", "DLR", "CSGP"],
    "⚙️ Utilities": ["NEE", "SO", "DUK", "SRE", "AEP", "PCG", "CEG", "D", "EXC", "XEL"]
}

# Define a function to display the welcome message and sector stocks
def show_welcome_message():
    st.markdown("""
    ## Da In Kim's 3rd Year Project
    ### Stock Price Prediction Using Ensemble Models
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify;">
    This dashboard presents the outcome of an ensemble model that combines time-series and deep learning models for stock price prediction. 
    <br><br>
    The ensemble model is crafted by integrating Prophet, ARIMA, LSTM, and GRU models, each weighted based on their Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE), and Directional Accuracy evaluated during the validation process. 
    <br><br>     
    The weights are then applied to the models to forecast future stock prices with improved accuracy.
    <br><br>
                
    - **Training set:** Data up to 2023/01/02
    - **Validation set:** 2023/01/03 to 2023/11/10
    - **Test set:** 2023/11/11 to 2024/02/02
                
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
    st.session_state.selected_ticker = 'Select a ticker'  # Reset selected ticker

selected_ticker = st.sidebar.selectbox('Choose a stock ticker:', ['Select a ticker'] + tickers, index=0)

if st.sidebar.button('Submit Ticker'):
    if selected_ticker and selected_ticker != 'Select a ticker':
        st.session_state.selected_ticker = selected_ticker
        st.session_state.home_page = False
    else:
        st.session_state.home_page = True

# Function to show model weights
def show_model_weights(ticker):
    if ticker in model_weights_df['TICKER'].values:
        weights = model_weights_df.loc[model_weights_df['TICKER'] == ticker, ['weight_prophet', 'weight_arima', 'weight_lstm', 'weight_gru']].squeeze()
        st.markdown(f"##### Model weights for {ticker}:", unsafe_allow_html=True)
        # Using st.metric to display each weight next to each other
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prophet", f"{weights['weight_prophet']:.2%}")
        with col2:
            st.metric("ARIMA", f"{weights['weight_arima']:.2%}")
        with col3:
            st.metric("LSTM", f"{weights['weight_lstm']:.2%}")
        with col4:
            st.metric("GRU", f"{weights['weight_gru']:.2%}")
        st.divider()
    else:
        st.write(f"No model weight information available for {ticker}")

# Function to load and show model performance metrics
def show_model_performance(ticker):
    st.write(f"##### Model performance metrics for {ticker}:", unsafe_allow_html=True)
    
    # Define the layout: one column for each model
    col_prophet, col_arima, col_lstm, col_gru = st.columns(4)
    model_names = ['Prophet', 'ARIMA', 'LSTM', 'GRU']
    metrics_columns = ['mape', 'mse', 'directional_accuracy']
    
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
                st.metric(label=f"{model_name} Directional Accuracy", value=f"{ticker_metrics['directional_accuracy']:.2f}%")
        
        except FileNotFoundError:
            st.error(f"The file for {model_name} could not be found. Please check the file path and name.")

# Function to calculate and display MAPE for the Ensemble model on the validation set
def calculate_and_display_mape(data, ticker):
    # Isolate the validation set and calculate MAPE
    validation_set = data[(data['Date'] >= "2023-01-03") & (data['Date'] <= "2023-11-10")]
    if not validation_set.empty and 'Ensemble' in validation_set.columns and 'Close' in validation_set.columns:
        mape = np.mean(np.abs((validation_set['Close'] - validation_set['Ensemble']) / validation_set['Close'])) * 100
        st.write(f"##### Ensemble Model MAPE for {ticker}: {mape:.2f}%", unsafe_allow_html=True)
        st.divider()
    else:
        st.write(f"No validation set data available for {ticker}")

# Function to load data and visualize it
def load_data(ticker):
    file_path = f'./ensemble/{ticker}_merged.csv'
    
    if ticker == 'Select a ticker':
        return

    try:
        # Load the data
        data = pd.read_csv(file_path)
        
        # Create Plotly graph object
        fig = go.Figure()

        # Ensure 'Close' and 'Ensemble' are displayed from the start
        for column in ['Close', 'Ensemble']:
            if column in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name=column, mode='lines'))
        
        # Add traces for model predictions, set them to be hidden in the graph initially
        for model in data.columns[2:]:  # Assuming the first column is Date and second is Close
            if model not in ['Close', 'Ensemble']:
                fig.add_trace(go.Scatter(x=data['Date'], y=data[model], name=model.upper(), mode='lines', visible='legendonly'))

        # Add shapes to highlight the validation and test set periods
        fig.add_vrect(
            x0="2023-01-03", x1="2023-11-10",
            annotation_text="Validation Set", annotation_position="top left",
            fillcolor="yellow", opacity=0.2, line_width=0
        )
        fig.add_vrect(
            x0="2023-11-11", x1="2024-02-02",
            annotation_text="Test Set", annotation_position="top left",
            fillcolor="blue", opacity=0.2, line_width=0
        )
        
        # Set graph layout
        fig.update_layout(title_text=f"{ticker} Stock Price Prediction", xaxis_title='Date', yaxis_title='Price')
        
        # Display the graph in the Streamlit app
        st.plotly_chart(fig)

        # Calculate and display MAPE for the Ensemble model on the validation set
        calculate_and_display_mape(data, ticker)

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