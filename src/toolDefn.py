import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import defaultdict, deque
import json
from google.generativeai.types import GenerationConfig
import google.generativeai as genai
import requests
from loguru import logger
from config import WHATSAPP_API_URL, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN


class ConversationMemory:
    """Manages conversation context for each user"""
    
    def __init__(self, max_messages_per_user: int = 10):
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_messages_per_user))
        self.max_messages = max_messages_per_user
    
    def add_message(self, user_id: str, user_message: str, bot_response: str):
        """Add a message exchange to conversation history"""
        self.conversations[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "bot": bot_response
        })
    
    def get_context(self, user_id: str, last_n: int = 3) -> str:
        """Get recent conversation context for a user"""
        if user_id not in self.conversations or not self.conversations[user_id]:
            return "No previous conversation"
        
        recent = list(self.conversations[user_id])[-last_n:]
        context_parts = []
        
        for msg in recent:
            context_parts.append(f"User: {msg['user']}")
            context_parts.append(f"Bot: {msg['bot']}")
        
        return "\n".join(context_parts)
    
    def get_history(self, user_id: str) -> List[Dict]:
        """Get full conversation history"""
        return list(self.conversations.get(user_id, []))
    
    def clear_user(self, user_id: str):
        """Clear conversation history for a user"""
        if user_id in self.conversations:
            del self.conversations[user_id]


class EmailCache:
    """Simple in-memory cache for email data"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: any):
        """Set cache value with timestamp"""
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
    
    def clear_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= timedelta(seconds=self.ttl)
        ]
        for key in expired_keys:
            del self.cache[key]


def read_file() -> Optional[int]:
    """Read history ID from file"""
    try:
        with open('../history_id.txt', 'r') as file:
            number_str = file.read().strip()
            return int(number_str) if number_str else None
    except FileNotFoundError:
        logger.warning("history_id.txt not found")
        return None
    except Exception as e:
        logger.error(f"Error reading history file: {e}")
        return None


def save_history_id(number: int):
    """Save history ID to file"""
    try:
        with open('../history_id.txt', 'w') as file:
            file.write(str(number))
    except Exception as e:
        logger.error(f"Error saving history ID: {e}")


def get_email_body(payload: dict) -> str:
    """Recursively extract email body from payload"""
    body = ""
    
    try:
        if "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain":
                    data = part.get("body", {}).get("data")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors='ignore')
                elif "parts" in part:
                    # Recurse into multipart
                    body = get_email_body(part)
                    if body:
                        return body
        
        # Handle non-multipart emails
        if payload.get('mimeType') == 'text/plain':
            data = payload.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error extracting email body: {e}")
        
    return body


def get_thread_content(service, thread_id: str) -> str:
    """Get full thread conversation content"""
    try:
        thread = service.users().threads().get(userId='me', id=thread_id).execute()
        messages = thread.get('messages', [])

        if not messages:
            return "Empty thread"

        full_conversation = []

        for msg in messages:
            headers = msg["payload"]["headers"]
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
            date = next((h["value"] for h in headers if h["name"] == "Date"), "Unknown")
            body = get_email_body(msg["payload"])

            if body:
                full_conversation.append(f"--- Message from {sender} on {date} ---\n{body}\n")
        
        return "\n".join(full_conversation) if full_conversation else "No content available"
    except Exception as e:
        logger.error(f"Error getting thread content: {e}")
        return f"Error retrieving thread: {str(e)}"


def summarize_content(content: str) -> str:
    """Summarize email content using AI"""
    if not content or not content.strip():
        return "No content to summarize."
    
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
        prompt = f"""Summarize this email in 2-3 concise sentences. Focus on:
1. Main topic/purpose
2. Key action items or questions
3. Any deadlines or urgency

Email content:
---
{content[:2000]}  # Limit to first 2000 chars
---

Summary:"""
        
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=150)
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return f"(Could not summarize) Snippet: {content[:200]}..."


def send_whatsapp_message(to: str, message: str) -> dict:
    """Send WhatsApp message with error handling and retries"""
    url = f"{WHATSAPP_API_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Truncate very long messages
    if len(message) > 4096:
        message = message[:4090] + "...\n\n(Message truncated)"
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.info(f"WhatsApp message sent to {to}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to send WhatsApp message after {max_retries} attempts: {str(e)}")


def format_email_for_whatsapp(email: dict) -> str:
    """Format email data for WhatsApp display"""
    from_addr = email.get('from', 'Unknown')
    subject = email.get('subject', 'No Subject')
    snippet = email.get('snippet', '')
    date = email.get('date', '')
    
    # Extract just the name or email
    if '<' in from_addr:
        from_name = from_addr.split('<')[0].strip()
    else:
        from_name = from_addr
    
    message = f"📧 *{subject}*\n"
    message += f"From: {from_name}\n"
    if date:
        message += f"Date: {date}\n"
    message += f"\n{snippet[:200]}"
    
    if len(snippet) > 200:
        message += "..."
    
    return message