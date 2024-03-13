# Stock Prediction Project

## Overview
This project encompasses a comprehensive approach to stock prediction utilizing a Jupyter notebook (`stock_prediction.ipynb`) for model training and validation, alongside a Streamlit-based GUI (`dashboard.py`) for displaying real-time forecast results. All files used in this project including datasets can be found at [https://drive.google.com/drive/folders/1oZlnt78J-hI4Z_nDl6OpbhtERlsKNM0m?usp=sharing](https://drive.google.com/drive/folders/1oZlnt78J-hI4Z_nDl6OpbhtERlsKNM0m?usp=sharing).

### stock_prediction.ipynb
The `stock_prediction.ipynb` notebook is designed to run in Google Colab. 
- This notebook serves as the backbone of the project, where various datasets are employed to train and validate the predictive models. 
- A key feature of this process is the computation of accuracy indices, which are crucial for determining the weights of individual models within the ensemble. 
- This approach ensures that the ensemble model is fine-tuned for optimal performance, providing a robust foundation for stock prediction.

### dashboard.py
The `dashboard.py` script utilizes Streamlit to create a user-friendly GUI, allowing users to interact with the predictive models in real-time. 
- By integrating the yfinance API, the dashboard fetches real-time stock data, enabling users to view forecast results over their desired time frames. 
- This interactive platform democratizes access to sophisticated stock forecasts, empowering users to make informed decisions based on the ensemble model's insights.
- Fashboard is deployed at [https://stock-prediction-dainkim.streamlit.app/](https://stock-prediction-dainkim.streamlit.app/).

