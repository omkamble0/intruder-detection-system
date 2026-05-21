import sqlite3

# Connect to the database
conn = sqlite3.connect("security_audit.db")
cursor = conn.cursor()

# Fetch all logs
cursor.execute("SELECT * FROM logs")
rows = cursor.fetchall()

print("--- SECURITY AUDIT LOGS ---")
for row in rows:
    print(f"ID: {row[0]} | Time: {row[1]} | Event: {row[2]}")
    print(f"Integrity Hash: {row[3]}\n")

conn.close()
