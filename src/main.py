# main.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API keys
groq_api_key = os.getenv("GROQ_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

if groq_api_key and serper_api_key:
    print("API keys loaded successfully!")
else:
    print("Error: API keys not loaded. Check your .env file.")

# You can print them to verify, but remove this line before sharing code
# print(f"Groq Key: {groq_api_key[:5]}...") 
# print(f"Serper Key: {serper_api_key[:5]}...")
