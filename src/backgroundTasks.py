from toolDefn import read_file, save_history_id, get_thread_content, send_whatsapp_message
from loguru import logger
from toolDefn import summarize_content
from services import get_email_service

def process_notification_in_background(start_history_id: str, new_history_id: int):
    service = get_email_service() 
    try:
        

        history_response = service.users().history().list(
            userId="me",
            startHistoryId=start_history_id
        ).execute()

        changes = history_response.get("history", [])
        
        newest_history_id = history_response.get('historyId', new_history_id)

        save_history_id(newest_history_id)
        logger.info(f"State updated to historyId: {newest_history_id}")

        if not changes:
            logger.info("No new history records found.")
            return {"status": "ok"}
        
        processed_threads = set()

        for record in changes:
            added_messages = record.get("messagesAdded", [])
            for item in added_messages:
                msg_id = item["message"]["id"]
                try:
                    # msg = service.users().messages().get(
                    #     userId="me", id=msg_id, format="full"
                    # ).execute()
                    msg_metadata = service.users().messages().get(
                        userId="me", id=msg_id, format="metadata"
                    ).execute()
                    
                    if 'INBOX' not in msg_metadata.get('labelIds', []):
                        continue 

                    thread_id = msg_metadata.get('threadId')

                    if thread_id and thread_id not in processed_threads:

                        processed_threads.add(thread_id)

                        headers = msg_metadata["payload"]["headers"]
                        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
                        # sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
                        # email_body = get_email_body(msg["payload"])

                        # if not email_body.strip():
                        #     continue 
                        full_thread_content = get_thread_content(service, thread_id)
                        if not full_thread_content.strip():
                            logger.warning(f"Thread {thread_id} has no text content to summarize.")
                            continue
                        thread_summary = summarize_content(full_thread_content)
                        notification_text = f"📧 New Reply in Thread: _{subject}_\n\n{thread_summary}"

                        # summary = summarize_content(f"Sender: {sender}, Subject: {subject}, Body:{email_body}")
                        send_whatsapp_message("918160376548", notification_text)
                
                except Exception as e:
                    logger.error(f"Failed to process individual message {msg_id}: {e}")
                    # This lets the loop continue even if one email fails
                    continue

    except Exception as e:
        logger.error(f"A major error occurred during history fetch: {e}")
        return {"status": "ok", "detail": str(e)}
