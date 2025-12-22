import sqlite3
import os
import sys
import shutil
from datetime import datetime

# Ensure we use the exact same path logic as app.py
# If /opt/render/project/src/vce_progress.db exists, use it.
RENDER_DB_PATH = '/opt/render/project/src/vce_progress.db'
if os.path.exists(RENDER_DB_PATH):
    DATABASE_PATH = RENDER_DB_PATH
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = os.path.join(BASE_DIR, 'vce_progress.db')

def migrate_db():
    print(f"Target Database: {DATABASE_PATH}")
    
    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Database not found at {DATABASE_PATH}")
        return

    # Backup
    backup_path = f"{DATABASE_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(DATABASE_PATH, backup_path)
        print(f"✅ Backup created at: {backup_path}")
    except Exception as e:
        print(f"⚠️  Backup failed: {e}")
        # Proceed with caution? No, let's stop.
        print("Aborting migration due to backup failure.")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if column exists first
        cursor.execute("PRAGMA table_info(question_attempts);")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'session_id' in columns:
            print("✅ Column 'session_id' already exists. No migration needed.")
        else:
            print("⏳ Adding 'session_id' column...")
            # Add the missing column
            cursor.execute("ALTER TABLE question_attempts ADD COLUMN session_id TEXT;")
            print("   Column added.")
            
            # Set a default value for existing records
            print("⏳ Backfilling existing records...")
            cursor.execute("UPDATE question_attempts SET session_id = 'legacy_session' WHERE session_id IS NULL;")
            print("   Records updated.")
            
            # Add index
            print("⏳ Creating index...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON question_attempts(session_id);")
            print("   Index created.")
            
            conn.commit()
            print("✅ Migration successful!")
            
        # Verify
        cursor.execute("PRAGMA table_info(question_attempts);")
        new_columns = [column[1] for column in cursor.fetchall()]
        if 'session_id' in new_columns:
            print("✅ Verification Passed: 'session_id' is present.")
        else:
            print("❌ Verification FAILED: 'session_id' is still missing.")
            
        conn.close()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("Restoring from backup...")
        try:
            shutil.copy2(backup_path, DATABASE_PATH)
            print("✅ Database restored.")
        except Exception as restore_error:
            print(f"❌ CRITICAL: Restore failed: {restore_error}")

if __name__ == "__main__":
    migrate_db()
