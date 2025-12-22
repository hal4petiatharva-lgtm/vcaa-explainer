import sqlite3
import os
import sys

# Ensure we use the exact same path logic as app.py
# If /opt/render/project/src/vce_progress.db exists, use it.
RENDER_DB_PATH = '/opt/render/project/src/vce_progress.db'
if os.path.exists(RENDER_DB_PATH):
    DATABASE_PATH = RENDER_DB_PATH
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = os.path.join(BASE_DIR, 'vce_progress.db')

def diagnose():
    print("="*50)
    print("DIAGNOSTIC REPORT")
    print("="*50)
    
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Script Location: {os.path.abspath(__file__)}")
    print(f"Computed DATABASE_PATH: {DATABASE_PATH}")
    
    if os.path.exists(DATABASE_PATH):
        print(f"✅ Database file found at: {DATABASE_PATH}")
        print(f"   Size: {os.path.getsize(DATABASE_PATH)} bytes")
    else:
        print(f"❌ Database file NOT found at: {DATABASE_PATH}")
        # Check if it exists in CWD
        cwd_db = os.path.join(os.getcwd(), 'vce_progress.db')
        if os.path.exists(cwd_db):
            print(f"⚠️  Found 'vce_progress.db' in CWD instead: {cwd_db}")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # List Tables
        print("\n--- Tables ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(tables)} tables: {', '.join(tables)}")
        
        # Check question_attempts Schema
        if 'question_attempts' in tables:
            print("\n--- Schema: question_attempts ---")
            cursor.execute("PRAGMA table_info(question_attempts);")
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]
            
            has_session_id = 'session_id' in columns
            
            for col in columns_info:
                print(f" - {col[1]} ({col[2]})")
                
            if has_session_id:
                print("\n✅ 'session_id' column EXISTS.")
                
                # Check sample data
                cursor.execute("SELECT id, session_id FROM question_attempts LIMIT 3")
                rows = cursor.fetchall()
                print("\n--- Sample Data (id, session_id) ---")
                for row in rows:
                    print(f" {row['id']}: {row['session_id']}")
            else:
                print("\n❌ 'session_id' column is MISSING.")
        else:
            print("\n❌ Table 'question_attempts' not found.")
            
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Error connecting/reading database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
