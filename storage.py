"""Small SQLite persistence layer for student profiles and practice history."""
import sqlite3
from pathlib import Path
import pandas as pd
DB_PATH = Path(__file__).parent / "data" / "coding_assistant.db"


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        con.execute("""CREATE TABLE IF NOT EXISTS practice_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, week INTEGER NOT NULL,
            topic TEXT NOT NULL, attempts INTEGER NOT NULL, problems_solved INTEGER NOT NULL,
            hints_used INTEGER NOT NULL, avg_time_min REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(student_id))""")


def create_student(student_id, display_name):
    with connect() as con:
        con.execute("INSERT INTO students(student_id, display_name) VALUES (?, ?)", (student_id, display_name))


def list_students():
    with connect() as con:
        return pd.read_sql_query("SELECT student_id, display_name FROM students ORDER BY display_name", con)


def add_practice(student_id, week, topic, attempts, solved, hints, avg_time):
    with connect() as con:
        con.execute("INSERT INTO practice_logs(student_id,week,topic,attempts,problems_solved,hints_used,avg_time_min) VALUES (?,?,?,?,?,?,?)",
                    (student_id, week, topic, attempts, solved, hints, avg_time))


def get_logs(student_id=None):
    with connect() as con:
        query = "SELECT student_id, week, topic, attempts, problems_solved, hints_used, avg_time_min FROM practice_logs"
        params = ()
        if student_id is not None:
            query += " WHERE student_id = ?"
            params = (student_id,)
        return pd.read_sql_query(query, con, params=params)


def import_logs(df):
    with connect() as con:
        for sid in df.student_id.astype(str).unique():
            con.execute("INSERT OR IGNORE INTO students(student_id, display_name) VALUES (?, ?)", (sid, sid))
        cols = ["student_id", "week", "topic", "attempts", "problems_solved", "hints_used", "avg_time_min"]
        df[cols].to_sql("practice_logs", con, if_exists="append", index=False)
