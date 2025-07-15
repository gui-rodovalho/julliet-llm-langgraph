import sqlite3
from faiss_vetorial import save_to_faiss
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
    if role == "user":
        save_to_faiss(content=content)

def ler_bd():
    conn = sqlite3.connect("conversas.db")
    cursor = conn.cursor()

    for row in cursor.execute("SELECT role, content, timestamp FROM historico"):
        print(row)
        
#ler_bd()
