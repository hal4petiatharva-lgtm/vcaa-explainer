import sqlite3

def migrate_db():
    try:
        conn = sqlite3.connect('vce_progress.db')
        cursor = conn.cursor()
        
        # Check if column exists first
        cursor.execute("PRAGMA table_info(question_attempts);")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'session_id' in columns:
            print("Column 'session_id' already exists. No migration needed.")
        else:
            print("Adding 'session_id' column...")
            # Add the missing column
            cursor.execute("ALTER TABLE question_attempts ADD COLUMN session_id TEXT;")
            
            # Set a default value for existing records
            print("Updating existing records...")
            cursor.execute("UPDATE question_attempts SET session_id = 'legacy_session' WHERE session_id IS NULL;")
            
            # Add index
            print("Creating index...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON question_attempts(session_id);")
            
            conn.commit()
            print("Migration successful!")
            
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate_db()
