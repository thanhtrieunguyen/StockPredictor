import streamlit as st
from data.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor
from models.decision_tree import DecisionTreeModel
from visualization.charts import ChartVisualizer
from data_manager import StockDataManager
from sklearn.model_selection import train_test_split
import pandas as pd
from models.knn import KNNModel
from models.linear_regression import LogisticRegressionModel
from models.naive_bayes import NaiveBayesModel
from models.pca import PCAModel
from models.random_forest import RandomForestModel 
import plotly.express as px
import numpy as np

def main():
    st.title('Hệ thống dự đoán giá cổ phiếu')
    
    # Thông số đầu vào
    symbol = st.text_input('Nhập Mã Cổ Phiếu (ví dụ: VFS):', 'VFS', key="symbol_input")
    prediction_days = st.selectbox('Chọn Thời Gian Dự Đoán (ngày):', 
                                   [1, 3, 7, 30, 90], key="timeframe_select", index=1)
    
    history_months = st.selectbox('Chọn Thời Gian Dữ Liệu Lịch Sử (tháng):', 
                                  [1, 2, 3, 6, 12], index=2)
    
    model_type = st.selectbox(
        'Chọn Mô hình dự đoán:',
        ['Decision Tree', 'Random Forest', 'KNN', 'Logistic Regression', 'Naive Bayes'],
        key="model_select"
    )

    # Tùy chọn sử dụng PCA để giảm chiều dữ liệu
    use_pca = st.checkbox('Sử dụng PCA để giảm chiều dữ liệu', value=False)
    if use_pca:
        variance_threshold = st.slider('Ngưỡng phương sai cho PCA', 0.8, 0.99, 0.95, 0.01)
    
    if st.button('Dự đoán'):
        # Khởi tạo quản lý dữ liệu và tải dữ liệu
        data_manager = StockDataManager()
        loader = DataLoader()
        
        # Cập nhật dữ liệu cổ phiếu
        with st.spinner('Đang cập nhật dữ liệu cổ phiếu...'):
            data = data_manager.update_stock_data(symbol, loader, history_months)
        
        if data is not None:
            # Tiền xử lý dữ liệu
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.create_features(data)
            X, y = preprocessor.prepare_data(processed_data, prediction_days)
            
            # Áp dụng PCA nếu được chọn
            if use_pca:
                pca_model = PCAModel(variance_threshold=variance_threshold)
                pca_model.train(X, y)
                X = pca_model.transform(X)
                
                # Hiển thị phân tích PCA
                st.subheader('Phân tích PCA')
                pca_info = pca_model.get_feature_importance()
                
                # Vẽ biểu đồ tỷ lệ phương sai giải thích tích lũy
                fig_pca = px.line(
                    y=np.cumsum(pca_info['explained_variance_ratio']),
                    title='Tỷ lệ phương sai tích lũy',
                    labels={'index': 'Số lượng thành phần', 'y': 'Tỷ lệ phương sai tích lũy'}
                )
                st.plotly_chart(fig_pca)
                
                st.write(f"Số lượng thành phần được chọn: {pca_info['n_components']}")
            
            # Chia tập dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Chọn và huấn luyện mô hình dựa trên lựa chọn của người dùng
            with st.spinner('Đang huấn luyện mô hình...'):
                if model_type == 'Decision Tree':
                    model = DecisionTreeModel()
                elif model_type == 'KNN':
                    model = KNNModel()
                elif model_type == 'Logistic Regression':
                    model = LogisticRegressionModel()
                elif model_type == 'Naive Bayes':
                    model = NaiveBayesModel()
                else:  # Random Forest
                    model = RandomForestModel()
                
                model.train(X_train, y_train)

            # Tạo dự đoán
            historical_predictions = pd.DataFrame(index=processed_data.index)
            historical_predictions['predicted_close'] = processed_data['close']
            historical_predictions['predicted_direction'] = model.predict(X)
            historical_predictions['actual_direction'] = y
            
            # Tính toán giá dự đoán lịch sử
            for i in range(len(historical_predictions)-1):
                if historical_predictions['predicted_direction'].iloc[i] == 1:
                    historical_predictions['predicted_close'].iloc[i+1] = \
                        historical_predictions['predicted_close'].iloc[i] * 1.01
                else:
                    historical_predictions['predicted_close'].iloc[i+1] = \
                        historical_predictions['predicted_close'].iloc[i] * 0.99
            
            # Tạo ngày và dự đoán tương lai
            last_date = processed_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='B'
            )
            
            # Tạo đặc trưng cho dự đoán tương lai
            last_record = processed_data.iloc[-1]
            future_features = pd.DataFrame(index=future_dates)
            for feature in X.columns if not use_pca else range(X.shape[1]):
                future_features[feature] = last_record[feature] if not use_pca else X[-1, feature]
            
            # Dự đoán tương lai
            future_predictions = model.predict(future_features)
            
            # Tính toán giá dự đoán tương lai
            last_price = processed_data['close'].iloc[-1]
            predicted_prices = [last_price]
            for pred in future_predictions:
                next_price = predicted_prices[-1] * (1.01 if pred == 1 else 0.99)
                predicted_prices.append(next_price)
            
            # Tạo DataFrame dự đoán
            pred_df = pd.DataFrame(index=future_dates)
            pred_df['predicted_close'] = predicted_prices[1:]
            
            # Hiển thị các chỉ số hiệu suất mô hình
            st.subheader('Các chỉ số hiệu suất của mô hình')
            accuracy = (model.predict(X_test) == y_test).mean()
            st.write(f'Độ chính xác trên tập kiểm tra: {accuracy:.2%}')
            
            # Hiển thị thông tin chi tiết theo mô hình
            if model_type == 'KNN':
                st.write('Các tham số tối ưu của mô hình:', model.get_best_params())
            elif model_type == 'Logistic Regression':
                feature_importance = model.get_feature_importance(
                    X.columns if not use_pca else [f'PC{i+1}' for i in range(X.shape[1])]
                )
                st.write('Độ quan trọng của các đặc trưng:', feature_importance)
            elif model_type == 'Naive Bayes':
                st.write('Xác suất tiên nghiệm:', model.evaluate_priors())
            elif model_type == 'Random Forest':
                feature_importance = model.get_feature_importance(
                    X.columns if not use_pca else [f'PC{i+1}' for i in range(X.shape[1])]
                )
                fig_importance = px.bar(
                    feature_importance,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Độ quan trọng của các đặc trưng'
                )
                st.plotly_chart(fig_importance)
                
            # Hiển thị kết quả
            chart = ChartVisualizer.create_stock_prediction_chart(
                processed_data.tail(30),
                pred_df,
                symbol,
                historical_predictions.tail(30)
            )
            
            st.plotly_chart(chart)
            
            # Hiển thị giá trị dự đoán
            st.subheader('Giá trị dự đoán')
            for date, price in pred_df.iterrows():
                st.write(f"{date.strftime('%Y-%m-%d')}: ${price['predicted_close']:.2f}")
            
            # Hiển thị thông tin dữ liệu
            st.subheader('Thông tin dữ liệu')
            st.write(f"Khoảng thời gian dữ liệu: {data.index.min().strftime('%Y-%m-%d')} đến {data.index.max().strftime('%Y-%m-%d')}")
            st.write(f"Tổng số ngày giao dịch: {len(data)}")

        else:
            st.error('Không thể lấy dữ liệu cổ phiếu. Vui lòng kiểm tra mã cổ phiếu và thử lại.')

if __name__ == "__main__":
    main()
