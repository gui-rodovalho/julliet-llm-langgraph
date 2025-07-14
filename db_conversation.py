import sqlite3

def save_msgs(role: str, content: str):
    conn = sqlite3.connect("conversas.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historico (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("INSERT INTO historico (role, content) VALUES (?, ?)", (role, content))
    conn.commit()