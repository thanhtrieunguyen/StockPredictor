import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ChartVisualizer:
    @staticmethod
    def create_stock_prediction_chart(historical_data, predictions, symbol, historical_predictions=None):
        """
        Create stock prediction chart with both historical and future predictions
        
        Parameters:
        historical_data: DataFrame with OHLCV data
        predictions: DataFrame with future predicted prices
        symbol: Stock symbol
        historical_predictions: DataFrame with historical predicted prices
        """
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f'{symbol} Stock Price & Predictions',
                'Prediction Accuracy',
                'Volume'
            )
        )

        # 1. Historical candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index,
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name='Historical Price'
            ),
            row=1, col=1
        )

        # 2. Add historical predictions if available
        if historical_predictions is not None:
            fig.add_trace(
                go.Scatter(
                    x=historical_predictions.index,
                    y=historical_predictions['predicted_close'],
                    mode='lines',
                    name='Historical Predictions',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )

            # Add prediction accuracy chart
            accuracy = (historical_predictions['predicted_direction'] == 
                       historical_predictions['actual_direction']).astype(int)
            
            fig.add_trace(
                go.Scatter(
                    x=historical_predictions.index,
                    y=accuracy.rolling(window=5).mean() * 100,
                    mode='lines',
                    name='5-Day Prediction Accuracy (%)',
                    line=dict(color='green'),
                ),
                row=2, col=1
            )

        # 3. Add future predictions
        if not predictions.empty:
            # Get last known price point
            last_date = historical_data.index[-1]
            last_price = historical_data['close'].iloc[-1]
            
            # Connect historical to prediction with a dotted line
            fig.add_trace(
                go.Scatter(
                    x=[last_date, predictions.index[0]],
                    y=[last_price, predictions['predicted_close'].iloc[0]],
                    mode='lines',
                    line=dict(color='red', dash='dot'),
                    name='Connection to Prediction'
                ),
                row=1, col=1
            )
            
            # Future prediction line
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['predicted_close'],
                    mode='lines+markers',
                    name='Future Predictions',
                    line=dict(color='red'),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )

        # 4. Volume chart
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['volume'],
                name='Volume',
                marker_color='rgba(0,0,0,0.5)'
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction Analysis',
            height=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            xaxis_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False
        )

        # Update axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickprefix="$", tickformat=".2f")
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1, ticksuffix="%")
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig