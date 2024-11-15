# Hướng dẫn sử dụng

## Bước 1: Cài đặt môi trường ảo Python

1. Mở cmd trong thư mục project
3. Tạo môi trường ảo bằng lệnh:

    python -m venv env

4. Kích hoạt môi trường ảo:
    - Trên Windows:
 
        .\env\Scripts\activate

    - Trên macOS và Linux:

        source env/bin/activate


## Bước 2: Cài đặt thư viện từ `requirements.txt`

1. Đảm bảo rằng bạn đang ở trong môi trường ảo.
2. Cài đặt các thư viện cần thiết bằng lệnh:

    pip install -r requirements.txt


## Bước 3: Chạy ứng dụng Streamlit

1. Đảm bảo rằng bạn đang ở trong môi trường ảo và đã cài đặt tất cả các thư viện cần thiết.
2. Chạy ứng dụng Streamlit bằng lệnh:

    streamlit run main.py

