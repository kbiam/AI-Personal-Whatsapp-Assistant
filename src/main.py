from contextlib import asynccontextmanager
from pyngrok import ngrok
import uvicorn
from fastapi import Body, FastAPI, Request, HTTPException, BackgroundTasks 
from fastapi.responses import PlainTextResponse
from loguru import logger
from datetime import datetime
import base64
from email.mime.text import MIMEText
from google import genai
from google.generativeai.types import GenerationConfig, FunctionDeclaration, Tool
import google.generativeai as genai
import json
from toolDefn import (
    read_file, 
    save_history_id, 
    get_email_body, 
    get_thread_content, 
    summarize_content, 
    send_whatsapp_message,
    ConversationMemory,
    EmailCache
)
from backgroundTasks import process_notification_in_background
from dotenv import load_dotenv
from services import get_email_service
from config import (
    CLIENT_ID,
    CLIENT_SECRET,
    REFRESH_TOKEN,
    NGROK_AUTH_TOKEN,
    APPLICATION_PORT,
    VERIFY_TOKEN,
    WHATSAPP_API_URL,
    WHATSAPP_PHONE_NUMBER_ID,
    WHATSAPP_ACCESS_TOKEN,
    GEMINI_API_KEY
)

load_dotenv()

genai.configure(api_key=GEMINI_API_KEY)

# Global services
gmail_service = None
conversation_memory = ConversationMemory()
email_cache = EmailCache(ttl_seconds=300)  # 5 minute cache

# Enhanced function declarations
list_emails_func = FunctionDeclaration(
    name="list_emails",
    description="Get a list of recent emails. Use this to find emails based on a query.",
    parameters={
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "The maximum number of emails to return. Defaults to 5."
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional query string to filter emails, following Gmail search syntax. "
                    "Examples: 'is:unread', 'from:example@gmail.com', 'subject:report', "
                    "'after:2024/01/01', 'has:attachment'"
                )
            },
            "fetch_full_content": {
                "type": "boolean",
                "description": "Whether to fetch full email body. Defaults to False for performance."
            },
            "fetch_thread": {
                "type": "boolean",
                "description": "Whether to fetch the full thread content. Defaults to False."
            }
        }
    }
)

send_email_func = FunctionDeclaration(
    name="send_email",
    description="Sends an email to a recipient.",
    parameters={
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "The email address of the recipient."
            },
            "subject": {
                "type": "string",
                "description": "The subject line of the email."
            },
            "body": {
                "type": "string",
                "description": "The main content/body of the email."
            }
        },
        "required": ["to", "subject", "body"]
    }
)

reply_to_email_func = FunctionDeclaration(
    name="reply_to_email",
    description="Reply to an existing email thread.",
    parameters={
        "type": "object",
        "properties": {
            "thread_id": {
                "type": "string",
                "description": "The thread ID to reply to."
            },
            "body": {
                "type": "string",
                "description": "The reply message body."
            }
        },
        "required": ["thread_id", "body"]
    }
)

search_emails_func = FunctionDeclaration(
    name="search_emails",
    description="Search emails with smart query construction. Use this for complex searches.",
    parameters={
        "type": "object",
        "properties": {
            "search_term": {
                "type": "string",
                "description": "Natural language search term to find emails."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results. Defaults to 10."
            }
        },
        "required": ["search_term"]
    }
)

# Create enhanced tool set
email_tools = Tool(function_declarations=[
    list_emails_func, 
    send_email_func, 
    reply_to_email_func,
    search_emails_func
])


def send_email(to: str, subject: str, body: str):
    """Send a new email"""
    global gmail_service
    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        
        send_result = gmail_service.users().messages().send(
            userId="me",
            body={"raw": raw_message}
        ).execute()
        
        logger.info(f"Email sent successfully to {to}")
        return {"status": "sent", "message_id": send_result["id"], "to": to, "subject": subject}
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise


def reply_to_email(thread_id: str, body: str):
    """Reply to an existing email thread"""
    global gmail_service
    try:
        # Get original thread to extract subject and recipient
        thread = gmail_service.users().threads().get(userId="me", id=thread_id).execute()
        messages = thread.get("messages", [])
        
        if not messages:
            raise ValueError("Thread not found or empty")
        
        # Get the first message headers
        original_msg = messages[0]
        headers = original_msg["payload"]["headers"]
        
        original_subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
        original_from = next((h["value"] for h in headers if h["name"] == "From"), "")
        
        # Extract email from "Name <email@domain.com>" format
        import re
        email_match = re.search(r'<(.+?)>', original_from)
        to_email = email_match.group(1) if email_match else original_from
        
        # Add Re: if not already present
        subject = original_subject if original_subject.startswith("Re:") else f"Re: {original_subject}"
        
        message = MIMEText(body)
        message["to"] = to_email
        message["subject"] = subject
        message["In-Reply-To"] = original_msg["id"]
        message["References"] = original_msg["id"]

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        
        send_result = gmail_service.users().messages().send(
            userId="me",
            body={"raw": raw_message, "threadId": thread_id}
        ).execute()
        
        logger.info(f"Reply sent to thread {thread_id}")
        return {"status": "replied", "message_id": send_result["id"], "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error replying to email: {e}")
        raise


def list_emails(max_results=5, query=None, fetch_full_content=False, fetch_thread=False):
    """Optimized email listing with caching and metadata-first approach"""
    global gmail_service, email_cache
    
    try:
        # Check cache first
        cache_key = f"list:{max_results}:{query}:{fetch_full_content}:{fetch_thread}"
        cached_result = email_cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached email list")
            return cached_result
        
        results = gmail_service.users().messages().list(
            userId="me",
            maxResults=max_results,
            q=query
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return []

        email_list = []

        def process_batch_response(request_id, response, exception):
            if exception is not None:
                logger.error(f"Error fetching message {request_id}: {exception}")
                return

            try:
                headers = response["payload"]["headers"]
                subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
                sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
                date = next((h["value"] for h in headers if h["name"] == "Date"), "Unknown")
                
                email_data = {
                    "id": response["id"],
                    "thread_id": response.get("threadId"),
                    "from": sender,
                    "subject": subject,
                    "date": date,
                    "snippet": response.get("snippet", ""),
                    "labels": response.get("labelIds", [])
                }

                # Only fetch body if explicitly requested
                if fetch_full_content:
                    email_data["body"] = get_email_body(response["payload"])

                # Only fetch thread if explicitly requested
                if fetch_thread:
                    thread_id = response.get("threadId")
                    if thread_id:
                        email_data["thread_content"] = get_thread_content(gmail_service, thread_id)

                email_list.append(email_data)
            except Exception as e:
                logger.error(f"Error processing email response: {e}")

        batch = gmail_service.new_batch_http_request(callback=process_batch_response)

        # Use metadata format for better performance
        format_type = "full" if fetch_full_content else "metadata"
        
        for msg in messages:
            batch.add(
                gmail_service.users().messages().get(
                    userId="me",
                    id=msg["id"],
                    format=format_type,
                    metadataHeaders=["Subject", "From", "Date", "To"]
                )
            )
        
        batch.execute()
        
        # Cache the result
        email_cache.set(cache_key, email_list)
        
        return email_list
    except Exception as e:
        logger.error(f"Error listing emails: {e}")
        raise


def search_emails(search_term: str, max_results=10):
    """Smart email search with query construction"""
    global gmail_service
    
    # Build smart query based on search term
    query_parts = []
    
    # Check for common keywords
    if "unread" in search_term.lower():
        query_parts.append("is:unread")
    if "important" in search_term.lower():
        query_parts.append("is:important")
    if "attachment" in search_term.lower():
        query_parts.append("has:attachment")
    
    # Add the search term itself
    query_parts.append(search_term)
    
    final_query = " ".join(query_parts)
    
    return list_emails(max_results=max_results, query=final_query, fetch_full_content=False)


AVAILABLE_FUNCTIONS = {
    "list_emails": list_emails,
    "send_email": send_email,
    "reply_to_email": reply_to_email,
    "search_emails": search_emails,
}


def process_user_request(message: str, user_id: str):
    """Enhanced request processing with conversation memory"""
    global conversation_memory
    
    # Get conversation context
    context = conversation_memory.get_context(user_id)
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        tools=[email_tools]
    )

    # Build prompt with context
    prompt = f"""You are an AI assistant that helps manage Gmail through natural conversation.

Previous context: {context}

User request: {message}

Instructions:
- Extract clear intent from the user's message
- Use appropriate tools to fulfill requests
- Be concise and helpful
- For email searches, intelligently construct Gmail search queries
- Consider the conversation context when responding
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.3)
        )

        # Handle function calls
        if response.candidates[0].content.parts[0].function_call:
            fn_call = response.candidates[0].content.parts[0].function_call
            fn_name = fn_call.name
            fn_args = dict(fn_call.args)

            logger.info(f"Calling function: {fn_name} with args: {fn_args}")

            if fn_name in AVAILABLE_FUNCTIONS:
                tool_result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

                # Generate human-friendly response
                final_response = model.generate_content(
                    f"""User asked: {message}

Tool '{fn_name}' returned:
{json.dumps(tool_result, indent=2)}

Respond naturally and concisely. Format for WhatsApp (use emojis sparingly, keep it brief).
Include key information like sender, subject, and date for emails.
Maximum 3-4 sentences unless listing multiple items."""
                )
                
                response_text = final_response.text
                
                # Store in conversation memory
                conversation_memory.add_message(user_id, message, response_text)
                
                return {"type": "message", "text": response_text, "function_used": fn_name}
        
        # Direct text response
        response_text = response.text
        conversation_memory.add_message(user_id, message, response_text)
        
        return {"type": "message", "text": response_text}
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"type": "error", "text": f"Sorry, I encountered an error: {str(e)}"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gmail_service
    logger.info("🚀 Starting Gmail-WhatsApp AI Agent")
    
    try:
        # Setup ngrok
        logger.info("Setting up ngrok tunnel...")
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(APPLICATION_PORT)
        logger.info(f"✅ Ngrok tunnel: {public_url}")

        # Initialize Gmail service
        gmail_service = get_email_service()
        logger.info("✅ Gmail service initialized")
        
        # Start watching Gmail
        watch_gmail()
        logger.info("✅ Gmail watch setup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

    yield
    
    logger.info("🛑 Shutting down...")
    ngrok.disconnect()


app = FastAPI(lifespan=lifespan, title="Gmail-WhatsApp AI Agent")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gmail_connected": gmail_service is not None
    }


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    """WhatsApp webhook verification"""
    mode = request.query_params.get('hub.mode')
    challenge = request.query_params.get('hub.challenge')
    token = request.query_params.get('hub.verify_token')
    
    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified")
        return challenge
    else:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/webhook")
async def receive_webhook(request: Request):
    """Handle incoming WhatsApp messages"""
    body = await request.json()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        changes = body.get("entry", [])[0].get("changes", [])[0]
        value = changes.get("value", {})
        
        if "messages" in value:
            message_data = value["messages"][0]
            message_text = message_data["text"]["body"]
            from_number = message_data["from"]
            
            logger.info(f"📨 Message from {from_number}: {message_text}")
            
            # Process with user context
            result = process_user_request(message=message_text, user_id=from_number)
            
            # Send response
            send_whatsapp_message(from_number, result["text"])
            logger.info(f"✅ Response sent to {from_number}")
        else:
            logger.debug("⚡ Non-message webhook event (status/ping)")

    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return {"status": "error", "detail": str(e)}
    
    return {"status": "ok"}


@app.post("/webhook/gmail/notification/push")
async def receive_push_notification_gmail(request: Request, background_tasks: BackgroundTasks):
    """Handle Gmail push notifications"""
    global gmail_service
    logger.info("📧 Gmail notification received")
    
    try:
        body = await request.json()

        if not body.get("message", {}).get("data"):
            logger.warning("Invalid Pub/Sub message format")
            return {"status": "ok"}

        payload = base64.urlsafe_b64decode(body["message"]["data"]).decode("utf-8")
        notification_data = json.loads(payload)
        new_history_id = notification_data['historyId']

        start_history_id = read_file()
        if not start_history_id:
            save_history_id(new_history_id)
            return {"status": "ok"}

        background_tasks.add_task(
            process_notification_in_background, 
            start_history_id, 
            new_history_id
        )

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"❌ Gmail notification error: {e}")
        return {"status": "error"}


@app.post("/send-email")
async def send_email_api(
    to: str = Body(...),
    subject: str = Body(...),
    body: str = Body(...)
):
    """API endpoint to send emails"""
    try:
        result = send_email(to, subject, body)
        return result
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-emails")
async def list_emails_api(
    max_results: int = 5, 
    query: str = None,
    fetch_full_content: bool = False
):
    """API endpoint to list emails"""
    try:
        emails = list_emails(max_results, query, fetch_full_content)
        return {"status": "success", "count": len(emails), "emails": emails}
    except Exception as e:
        logger.error(f"Error listing emails: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_ai(request: Request):
    """API endpoint for AI chat"""
    body = await request.json()
    message = body.get("message")
    user_id = body.get("user_id", "api_user")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        result = process_user_request(message, user_id)
        return {"status": "success", "response": result}
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{user_id}")
async def get_conversation_history(user_id: str):
    """Get conversation history for a user"""
    history = conversation_memory.get_history(user_id)
    return {"user_id": user_id, "history": history}


@app.delete("/conversation/{user_id}")
async def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    conversation_memory.clear_user(user_id)
    return {"status": "cleared", "user_id": user_id}


def watch_gmail():
    """Setup Gmail push notifications"""
    global gmail_service
    try:
        request_body = {
            'labelIds': ['INBOX'],
            'topicName': 'projects/whatsapp-ai-chatbot-473718/topics/email',
            'labelFilterBehavior': 'INCLUDE'
        }
        response = gmail_service.users().watch(userId="me", body=request_body).execute()
        save_history_id(response['historyId'])
        logger.info(f"Gmail watch setup: {response}")
    except Exception as e:
        logger.error(f"Error setting up Gmail watch: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=APPLICATION_PORT, 
        reload=True,
        log_level="info"
    )