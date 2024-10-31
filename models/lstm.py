import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10  # Number of time steps to look back
        
    def create_sequences(self, X, y=None):
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            seq = X[i:(i + self.sequence_length)]
            if len(seq) == self.sequence_length:
                sequences.append(seq)
                if y is not None:
                    targets.append(y.iloc[i + self.sequence_length])

        
        return np.array(sequences), np.array(targets)

    def train(self, X_train, y_train):
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_train)
        
        # Ensure X_seq and y_seq have the same length
        min_length = min(len(X_seq), len(y_seq))
        X_seq = X_seq[:min_length]
        y_seq = y_seq[:min_length]
        
        # Define model architecture
        self.model = Sequential([
            Input(shape=(self.sequence_length, X_train.shape[1])),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        batch_size = 32
        num_batches = len(X_seq) // batch_size
        
        for epoch in range(50):
            epoch_loss = 0
            for i in range(0, len(X_seq), batch_size):
                batch_x = X_seq[i:i+batch_size]
                batch_y = y_seq[i:i+batch_size]
                loss, _ = self.train_step(batch_x, batch_y)
                epoch_loss += loss
            print(f"Epoch {epoch+1}/{50}, Loss: {epoch_loss/num_batches:.4f}")
            
        return self.model

    @tf.function(reduce_retracing=True)
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = tf.keras.losses.MeanSquaredError()(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred
    
    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            # Handle case where X is shorter than sequence length
            pad_length = self.sequence_length - len(X)
            if pad_length > 0:
                padding = np.zeros((pad_length, X.shape[1]))
                X_scaled = np.vstack([padding, X_scaled])
            X_seq = np.array([X_scaled])
        
        # Get predictions
        predictions = self.model.predict(X_seq)
        
        return predictions.flatten()
    
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