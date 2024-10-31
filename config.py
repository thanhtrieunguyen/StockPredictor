import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
if not TWELVE_DATA_API_KEY:
    raise ValueError("Twelve Data API key not found in environment variables. "
                    "Please add your API key to the .env file.")

BASE_URL = 'https://api.twelvedata.com'

# Validate API key format
def validate_api_key():
    # Twelve Data API keys are 32 characters long
    if len(str(TWELVE_DATA_API_KEY)) != 32:
        raise ValueError("Invalid API key format. Twelve Data API keys should be 32 characters long.")
    
validate_api_key()