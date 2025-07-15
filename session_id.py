import sqlite3

def get_next_session_id(table):
    #table = "julliet_dupo"
    conn = sqlite3.connect('session_id.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
    # Recupera o último session_id
    
    result = cursor.fetchone()
    if result:
        cursor.execute(f'SELECT MAX(CAST(session_id AS INTEGER)) FROM {table}')
        resultado = cursor.fetchone()
        last_session_id = resultado[0] if resultado[0] is not None else 0
        
        # Incrementa para o próximo session_id
        next_session_id = int(last_session_id) + 1

        print(f"esse é o novo session id - {next_session_id}")
        # Insere o novo session_id no banco de dados
        
        conn.close()
        novo_id(next_session_id, table)
    
    else: 
        # Cria a tabela caso ela não exista
        next_session_id = 1
        cursor.execute(f'''
            CREATE TABLE {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER
            );
        ''')
        novo_id(next_session_id, table)

    
    return next_session_id

def novo_id(next_session_id, table):
    print("novo db")
    conn = sqlite3.connect('session_id.db')
    
    cursor = conn.cursor()
    

    cursor.execute(f'''
        INSERT INTO {table} (session_id) VALUES (?)
    ''', (next_session_id,))
    conn.commit()

#get_next_session_id()