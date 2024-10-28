import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Alpha Vantage API key not found in environment variables. "
                    "Please add your API key to the .env file.")

BASE_URL = 'https://www.alphavantage.co/query'

# Validate API key format
def validate_api_key():
    if len(str(ALPHA_VANTAGE_API_KEY)) != 16:
        raise ValueError("Invalid API key format. Alpha Vantage API keys should be 16 characters long.")
    
validate_api_key()