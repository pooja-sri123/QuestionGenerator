import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT,
        score INTEGER,
        accuracy REAL,
        avg_time REAL,
        difficulty TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def save_session(student, score, accuracy, avg_time, difficulty):
    conn = sqlite3.connect('database.db')
    conn.execute(
        "INSERT INTO sessions VALUES (NULL,?,?,?,?,?,datetime('now'))",
        (student, score, accuracy, avg_time, difficulty))
    conn.commit()
    conn.close()

def get_sessions():
    conn = sqlite3.connect('database.db')
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return rows