#!/usr/bin/env python3
"""
Check recent scraper logs in the database.
"""

import os
from supabase import create_client

# Set up environment
os.environ['SUPABASE_URL'] = 'https://gjakiqtayqltssvbzasd.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdqYWtpcXRheXFsdHNzdmJ6YXNkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjkxOTU4OCwiZXhwIjoyMDU4NDk1NTg4fQ.wY8akPd9J-aRVQAOwTiFuOPxWM90fvkvXpyEfPogyfw'

def check_logs():
    """Check recent scraper logs."""
    client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    
    print("Recent scraper logs:")
    print("=" * 80)
    
    try:
        result = client.table('scraper_logs').select('*').order('start_time', desc=True).limit(5).execute()
        
        if result.data:
            for i, log in enumerate(result.data, 1):
                print(f"Run {i}:")
                print(f"  ID: {log['run_id'][:8]}...")
                print(f"  Status: {log['status']}")
                print(f"  Start: {log['start_time']}")
                print(f"  End: {log['end_time']}")
                print(f"  Records Found: {log['total_records_found']}")
                print(f"  Records Processed: {log['new_records_processed']}")
                print(f"  Years: {log['years_processed']}")
                print(f"  Execution Time: {log['execution_time_seconds']}s")
                if log['error_message']:
                    print(f"  Error: {log['error_message']}")
                print("-" * 40)
        else:
            print("No logs found in database.")
            
    except Exception as e:
        print(f"Error checking logs: {e}")

if __name__ == "__main__":
    check_logs()
