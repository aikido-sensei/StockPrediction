# LTSM Stock Price Prediction

## Project Description
This project involves using an LTSM (Long Short-Term Memory) model to predict the closing prices of a stock, equipped with data fetching capabilities to retrieve the required data automatically. Although it is specifically configured for Google (GOOG), it can be easily adapted for other stocks.

## Features
- Automated data fetching from the alpha vantage API.
- Data processing for time series data.
- An LTSM model built using Keras for stock price predictions.
- Data normalization with MinMaxScaler from scikit-learn.
- Evaluation of model predictions using Mean Squared Error (MSE).
- Methods to inverse transform predictions to the original scale.
