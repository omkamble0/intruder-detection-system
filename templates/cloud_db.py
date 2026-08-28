# cloud_db.py
from supabase import create_client, Client

SUPABASE_URL = "https://raghqovbiobywbjbykmj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhZ2hxb3ZiaW9ieXdiamJ5a21qIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkzNjg2MzUsImV4cCI6MjA5NDk0NDYzNX0.1keZ2Ku1nyVSga_XGiVUkvZ4qexYHM3vJnKznrE-_v4"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_activity(user_id, tool, details):
    """Logs actions to the cloud database."""
    if not user_id or user_id == "guest":
        return
    try:
        supabase.table('activity_logs').insert({
            "user_id": user_id, 
            "tool_used": tool, 
            "action_details": details
        }).execute()
    except Exception as e:
        print(f"Cloud Logging Error: {e}")
        