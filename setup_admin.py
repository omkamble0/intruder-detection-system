import sqlite3
import hashlib
import settings

def create_admin():
    conn = sqlite3.connect(settings.DATABASE_NAME)
    cursor = conn.cursor()
    
    # Create the users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT)''')
    
    # Setup your default credentials
    username = "admin"
    password = "password123" # You can change this
    
    # Hash the password for enterprise-grade security
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        print(f"✅ SUCCESS: Admin user '{username}' created.")
    except sqlite3.IntegrityError:
        print("⚠️ User 'admin' already exists in the database.")
        
    conn.close()

if __name__ == "__main__":
    create_admin()
    