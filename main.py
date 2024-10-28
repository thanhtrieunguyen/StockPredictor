import streamlit as st
from data.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor
from models.decision_tree import DecisionTreeModel
from visualization.charts import ChartVisualizer
from data_manager import StockDataManager
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    st.title('Stock Price Prediction App')
    
    # Input parameters
    symbol = st.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL', key="symbol_input")
    prediction_days = st.selectbox('Select Prediction Timeframe (days):', 
                                   [1, 3, 7, 30, 90], key="timeframe_select")
    
    history_months = st.selectbox('Select Historical Data Period (months):', 
                                [1, 2, 3, 6, 12], index=2)
    
    # Update stock data

            
    # if st.button('Predict', key="predict_button"):
    #     # Load data
    #     loader = DataLoader()
    #     data = loader.get_stock_data(symbol)
        
    #     with st.spinner('Updating stock data...'):
    #         data = data_manager.update_stock_data(symbol, loader, history_months)

    #     if data is not None:
    #         # Preprocess data
    #         preprocessor = DataPreprocessor()
    #         processed_data = preprocessor.create_features(data)
    #         X, y = preprocessor.prepare_data(processed_data, prediction_days)
            
    #         # Split data
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=0.2, shuffle=False
    #         )
            
    #         # Train model
    #         with st.spinner('Training model...'):
    #                 model = DecisionTreeModel()
    #                 model.train(X_train, y_train)
        
    #     # Generate future dates
    #         last_date = processed_data.index[-1]
    #         future_dates = pd.date_range(
    #             start=last_date + pd.Timedelta(days=1),
    #             periods=prediction_days,
    #             freq='B'
    #         )
        
    #     # Create features for future prediction
    #     last_record = processed_data.iloc[-1]
    #     future_features = pd.DataFrame(index=future_dates)
        
    #     # Initialize with last known values
    #     for feature in X.columns:
    #         future_features[feature] = last_record[feature]
        
    #     # Make predictions for future dates
    #     future_predictions = model.predict(future_features)
        
    #     # Calculate predicted prices
    #     last_price = processed_data['close'].iloc[-1]
    #     predicted_prices = [last_price]
        
    #     for pred in future_predictions:
    #         if pred == 1:
    #             next_price = predicted_prices[-1] * 1.01  # Increase 1%
    #         else:
    #             next_price = predicted_prices[-1] * 0.99  # Decrease 1%
    #         predicted_prices.append(next_price)
        
    #     # Create prediction DataFrame
    #     pred_df = pd.DataFrame(index=future_dates)
    #     pred_df['predicted_close'] = predicted_prices[1:]  # Skip first value (last actual price)
        
    #     # Visualize results
    #     chart = ChartVisualizer.create_stock_prediction_chart(
    #         processed_data, pred_df, symbol
    #     )
    #     st.plotly_chart(chart)
        
    #     # Show metrics
    #     accuracy = (model.predict(X_test) == y_test).mean()
    #     st.write(f'Model Accuracy: {accuracy:.2%}')
        
    #     # Show predicted values
    #     st.subheader('Predicted Prices')
    #     for date, price in pred_df.iterrows():
    #         st.write(f"{date.strftime('%Y-%m-%d')}: ${price['predicted_close']:.2f}")
    # st.title('Stock Price Prediction App')
    
    # # Input parameters
    # symbol = st.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL')
    # prediction_days = st.selectbox('Select Prediction Timeframe:', 
    #                              [1, 3, 7, 30, 90])
    
    if st.button('Predict'):
        # Initialize data manager and loader
        data_manager = StockDataManager()
        loader = DataLoader()
        
        # Update stock data
        with st.spinner('Updating stock data...'):
            data = data_manager.update_stock_data(symbol, loader, history_months)
        
        if data is not None:
            # Preprocess data
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.create_features(data)
            X, y = preprocessor.prepare_data(processed_data, prediction_days)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Train model
            with st.spinner('Training model...'):
                model = DecisionTreeModel()
                model.train(X_train, y_train)
            
            # Generate future dates and predictions
            last_date = processed_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='B'
            )
            
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
            
            # Visualize results
            chart = ChartVisualizer.create_stock_prediction_chart(
                processed_data.tail(30),  # Show last 30 days
                pred_df,
                symbol
            )
            st.plotly_chart(chart)
            
            # Show metrics
            accuracy = (model.predict(X_test) == y_test).mean()
            st.write(f'Model Accuracy: {accuracy:.2%}')
            
            # Show predicted values
            st.subheader('Predicted Prices')
            for date, price in pred_df.iterrows():
                st.write(f"{date.strftime('%Y-%m-%d')}: ${price['predicted_close']:.2f}")
            
            # Show data info
            st.subheader('Data Information')
            st.write(f"Data range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
            st.write(f"Total trading days: {len(data)}")
        else:
            st.error('Failed to retrieve stock data. Please check the symbol and try again.')

if __name__ == "__main__":
    main()