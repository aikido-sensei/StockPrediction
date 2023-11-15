import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from keras.optimizers import Adam, AdamW, Adadelta
import matplotlib.pyplot as plt
import numpy as np


class LTSM:

    def __init__(self, batch_size):
        self.model = None
        self.batch_size = batch_size
        self.train_data, self.test_data, self.scaler_used = self.get_processed_data()

    def get_formatted_data(self):
        file_path = '../DataFetching/GOOG-2023-10-18.json'

        # Load the JSON file directly
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract the "Time Series (Daily)" data
        time_series_data = data['Time Series (Daily)']

        # Load it into a DataFrame
        df = pd.DataFrame.from_dict(time_series_data, orient='index')

        return df

    def get_processed_data(self):
        df = self.get_formatted_data()

        # Convert data types to float
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Drop rows with NaN values that might have resulted from conversion
        df = df.dropna()

        # Normalize the entire data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        # Create a separate scaler for the closing prices only
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(df[["4. close"]])

        # Create sequences and labels
        data_pairs = []

        for i in range(0, len(scaled_data) - self.batch_size, self.batch_size):
            X = scaled_data[i:i + self.batch_size]
            if i + self.batch_size < len(scaled_data):
                y = scaled_data[i + self.batch_size, df.columns.get_loc("4. close")]
                data_pairs.append((X, y))

        train_size = int(0.8 * len(data_pairs))
        train_data = data_pairs[:train_size]
        test_data = data_pairs[train_size:]

        return train_data, test_data, close_scaler


    def build_model(self, input_shape):
        model = Sequential()

        model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))

        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=100))
        model.add(Dropout(0.6))

        model.add(Dense(units=1))
        optimizer = AdamW(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        self.model = model

    def train_model(self, epochs):
        X_train, y_train = zip(*self.train_data)  # Use train_data
        X_train, y_train = np.array(X_train), np.array(y_train)

        input_shape = (X_train.shape[1], X_train.shape[2])  # Get the input shape
        self.build_model(input_shape)  # Build the model with the given input shape

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)

        # Modify the training process to return the training history
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size, validation_split=0.1)

        # Plot the training loss and validation loss
        self.plot_loss(history)

    def plot_loss(self, history):
        plt.figure(figsize=(12, 6))

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def inverse_transform_predictions(self, predictions):
        return self.scaler_used.inverse_transform(predictions.reshape(-1, 1))

    def test_model(self):
        # Separate features and labels from test_data
        X_test, y_test = zip(*self.test_data)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Make predictions
        predictions = self.model.predict(X_test)

        # Inverse transform the predictions and y_test to original scale
        original_scale_predictions = self.inverse_transform_predictions(predictions)
        original_scale_y_test = self.scaler_used.inverse_transform(y_test.reshape(-1, 1))

        # Compute the mean squared error or any other error metric
        mse = np.mean((original_scale_predictions - original_scale_y_test) ** 2)
        print(f"Mean Squared Error on Test Data: {mse}")


ltsm = LTSM(batch_size=10)

# Train the model for a specified number of epochs
ltsm.train_model(epochs=25)

# Test the model
ltsm.test_model()