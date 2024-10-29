import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10  # Number of time steps to look back
        
    def create_sequences(self, X, y=None):
        """Create sequences for LSTM input"""
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            seq = X[i:(i + self.sequence_length)]
            sequences.append(seq)
            if y is not None:
                targets.append(y[i + self.sequence_length])
                
        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences)
    
    def train(self, X_train, y_train):
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_train)
        
        # Define model architecture
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, 
                 input_shape=(self.sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.model
    
    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            # Handle case where X is shorter than sequence length
            # Pad with zeros or last known values
            pad_length = self.sequence_length - len(X)
            if pad_length > 0:
                padding = np.zeros((pad_length, X.shape[1]))
                X_scaled = np.vstack([padding, X_scaled])
            X_seq = np.array([X_scaled])
        
        # Get predictions
        predictions = self.model.predict(X_seq)
        
        # Convert probabilities to binary predictions
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            pad_length = self.sequence_length - len(X)
            if pad_length > 0:
                padding = np.zeros((pad_length, X.shape[1]))
                X_scaled = np.vstack([padding, X_scaled])
            X_seq = np.array([X_scaled])
        
        # Get prediction probabilities
        probas = self.model.predict(X_seq)
        
        # Convert to 2D array of probabilities for both classes
        return np.hstack([1-probas, probas])