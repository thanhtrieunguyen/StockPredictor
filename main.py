import streamlit as st
from data.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor
from models.decision_tree import DecisionTreeModel
from visualization.charts import ChartVisualizer
from data_manager import StockDataManager
from models.knn import KNNModel
from models.linear_regression import LogisticRegressionModel
from models.naive_bayes import NaiveBayesModel
from models.pca import PCAModel
from models.lstm import LSTMModel
from models.random_forest import RandomForestModel
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def main():
    st.title('Stock Price Prediction App')
    
    # Input parameters
    symbol = st.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL', key="symbol_input")
    prediction_days = st.selectbox('Select Prediction Timeframe (days):', 
                                   [1, 3, 7, 30, 90], key="timeframe_select")
    
    history_months = st.selectbox('Select Historical Data Period (months):', 
                                [1, 2, 3, 6, 12], index=2)
    
    # Add model selection
    model_type = st.selectbox(
        'Select Machine Learning Model:',
        ['Decision Tree', 'KNN', 'Logistic Regression', 'Naive Bayes', 'PCA', 'LSTM', 'Random Forest'],
        key="model_select"
    )

    if st.button('Predict'):
        try:
            # Initialize data manager and loader
            data_manager = StockDataManager()
            loader = DataLoader()
            
            # Update stock data
            with st.spinner('Updating stock data...'):
                data = data_manager.update_stock_data(symbol, loader, history_months)
            
            if data is not None and len(data) > prediction_days:
                # Preprocess data
                preprocessor = DataPreprocessor()
                processed_data = preprocessor.create_features(data)
                X, y = preprocessor.prepare_data(processed_data, prediction_days)
                
                # Validate data size
                if len(X) < 2:
                    st.error("Not enough data points for prediction. Please increase the historical data period.")
                    return
                
                # Determine appropriate test size based on data length
                min_test_size = max(0.1, 2/len(X))  # Ensure at least 2 samples in test set
                max_test_size = min(0.3, 1 - 2/len(X))  # Ensure at least 2 samples in train set
                test_size = min(max_test_size, max(min_test_size, 0.2))  # Default to 0.2 if possible
                
                # Split data with adjusted test size
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=False
                )
                
                # Validate split results
                if len(X_train) < 2 or len(X_test) < 2:
                    st.error(f"Insufficient data for training. Need at least 4 data points, but got {len(X)}.")
                    return
                
                # Select and train model based on user choice
                with st.spinner('Training model...'):
                    if model_type == 'Decision Tree':
                        model = DecisionTreeModel()
                    elif model_type == 'KNN':
                        model = KNNModel()
                    elif model_type == 'Logistic Regression':
                        model = LogisticRegressionModel()
                    elif model_type == 'Naive Bayes':
                        model = NaiveBayesModel()
                    elif model_type == 'PCA':
                        model = PCAModel()
                    elif model_type == 'LSTM':
                        model = LSTMModel()
                    else:  # Random Forest
                        model = RandomForestModel()
                    
                    model.train(X_train, y_train)
                
                # Generate future dates and predictions
                last_date = processed_data.index[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=prediction_days,
                    freq='B'
                )

                # Generate historical predictions
                historical_predictions = pd.DataFrame(index=processed_data.index)
                historical_predictions['predicted_close'] = processed_data['close']
                
                historical_predictions['predicted_direction'] = model.predict(X)
                historical_predictions['actual_direction'] = y
                
                # Calculate historical predicted prices
                for i in range(len(historical_predictions)-1):
                    if historical_predictions['predicted_direction'].iloc[i] == 1:
                        historical_predictions['predicted_close'].iloc[i+1] = \
                            historical_predictions['predicted_close'].iloc[i] * 1.01
                    else:
                        historical_predictions['predicted_close'].iloc[i+1] = \
                            historical_predictions['predicted_close'].iloc[i] * 0.99
                
                # Create features for future prediction
                last_record = processed_data.iloc[-1]
                future_features = pd.DataFrame(index=future_dates)
                for feature in X.columns:
                    future_features[feature] = last_record[feature]
                
                # Make predictions
                future_predictions = model.predict(future_features)
                
                # Calculate predicted prices
                last_price = processed_data['close'].iloc[-1]
                predicted_prices = [last_price]
                for pred in future_predictions:
                    next_price = predicted_prices[-1] * (1.01 if pred == 1 else 0.99)
                    predicted_prices.append(next_price)
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame(index=future_dates)
                pred_df['predicted_close'] = predicted_prices[1:]
                
                # Show comparative metrics
                st.subheader('Model Performance Metrics')
                accuracy = (model.predict(X_test) == y_test).mean()
                st.write(f'Test Set Accuracy: {accuracy:.2%}')
                
                if hasattr(model.model, 'best_params_'):
                    st.write('Best Model Parameters:')
                    st.json(model.model.best_params_)

                if model_type == 'Random Forest':
                    # Show feature importance
                    feature_importance = model.get_feature_importance(X.columns)
                    fig_importance = px.bar(
                        feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Feature Importance in Random Forest Model'
                    )
                    st.plotly_chart(fig_importance)
                
                elif model_type == 'LSTM':
                    # Show training history if available
                    if hasattr(model.model, 'history'):
                        history = model.model.history.history
                        metrics_df = pd.DataFrame(history)
                        fig_metrics = px.line(
                            metrics_df,
                            title='LSTM Training History',
                            labels={'value': 'Value', 'index': 'Epoch'}
                        )
                        st.plotly_chart(fig_metrics)
                
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual Down', 'Actual Up'],
                    columns=['Predicted Down', 'Predicted Up']
                )
                
                st.subheader('Confusion Matrix')
                st.dataframe(cm_df)

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                st.write(f'Precision: {precision:.2%}')
                st.write(f'Recall: {recall:.2%}')
                st.write(f'F1 Score: {f1:.2%}')

                chart = ChartVisualizer.create_stock_prediction_chart(
                    processed_data.tail(30),
                    pred_df,
                    symbol,
                    historical_predictions.tail(30)
                )
                st.plotly_chart(chart)

                # Show predicted values
                st.subheader('Predicted Prices')
                for date, price in pred_df.iterrows():
                    st.write(f"{date.strftime('%Y-%m-%d')}: ${price['predicted_close']:.2f}")
                
                # Show data info
                st.subheader('Data Information')
                st.write(f"Data range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
                st.write(f"Total trading days: {len(data)}")
            else:
                st.error('Insufficient data for prediction. Please check the symbol and try again.')
                
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
            st.error('Please try again with different parameters or contact support.')

if __name__ == "__main__":
    main()