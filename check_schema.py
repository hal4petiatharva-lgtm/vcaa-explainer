import sqlite3

def check_schema():
    try:
        conn = sqlite3.connect('vce_progress.db')
        cursor = conn.execute("PRAGMA table_info(question_attempts);")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Columns in question_attempts: {columns}")
        
        if 'session_id' in columns:
            print("SUCCESS: 'session_id' column exists.")
        else:
            print("FAILURE: 'session_id' column is MISSING.")
            
        conn.close()
    except Exception as e:
        print(f"Error checking schema: {e}")

if __name__ == "__main__":
    check_schema()
