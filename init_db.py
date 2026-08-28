import sqlite3
import hashlib
import settings

def initialize_database():
    conn = sqlite3.connect(settings.DATABASE_NAME)
    cursor = conn.cursor()

    # 1. Create the Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')

    # 2. Create the Activity Logs table (Linked via Foreign Key)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tool_used TEXT NOT NULL,
            action_details TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Multi-tenant database initialized securely.")

if __name__ == "__main__":
    initialize_database()