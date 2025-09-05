import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env file!")

model_name = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# LLM client
client = AsyncOpenAI(base_url=BASE_URL, api_key=GEMINI_API_KEY)