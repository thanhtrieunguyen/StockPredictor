import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ChartVisualizer:
    @staticmethod
    def create_stock_prediction_chart(historical_data, predictions, symbol):
        """Create stock prediction chart with future predictions"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Stock Price', 'Volume'),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # Historical candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index,
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name='Historical'
            ),
            row=1, col=1
        )

        # Add last known price point
        last_date = historical_data.index[-1]
        last_price = historical_data['close'].iloc[-1]
        
        # Predicted values with confidence interval
        if not predictions.empty:
            # Connect historical to prediction with a dotted line
            fig.add_trace(
                go.Scatter(
                    x=[last_date, predictions.index[0]],
                    y=[last_price, predictions['predicted_close'].iloc[0]],
                    mode='lines',
                    line=dict(color='red', dash='dot'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Future predictions
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['predicted_close'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red'),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )

        # Volume chart
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['volume'],
                name='Volume'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction',
            yaxis_title='Price ($)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickprefix="$", tickformat=".2f")
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig
    @staticmethod
    def create_stock_prediction_chart(historical_data, predictions, symbol):
        """
        Tạo biểu đồ dự đoán giá chứng khoán
        """
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=(f'{symbol} Stock Price', 'Volume'))

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=historical_data.index,
                open=historical_data['open'],
                high=historical_data['high'],
                low=historical_data['low'],
                close=historical_data['close'],
                name='Historical'
            ),
            row=1, col=1
        )

        # Predicted values
        if predictions is not None:
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions['predicted_close'],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )

        # Volume chart
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['volume'],
                name='Volume'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig