import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env file
load_dotenv()

# Get API key
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Select model
model = genai.GenerativeModel("gemini-2.5-flash")

# Send prompt
response = model.generate_content(
    "Explain what a product manager does in simple words"
)

print(response.text)
