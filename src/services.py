import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from config import (
    CLIENT_ID,
    CLIENT_SECRET,
    REFRESH_TOKEN,
    NGROK_AUTH_TOKEN,
    APPLICATION_PORT,
    VERIFY_TOKEN,
    WHATSAPP_API_URL,
    WHATSAPP_PHONE_NUMBER_ID,
    WHATSAPP_ACCESS_TOKEN
)

def get_email_service():
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://mail.google.com/"]

    )
    creds.refresh(GoogleRequest())
    service = build('gmail', 'v1', credentials=creds)
    return service
