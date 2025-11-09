import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
APPLICATION_PORT = os.getenv("APPLICATION_PORT")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")


# whatsapp config
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")