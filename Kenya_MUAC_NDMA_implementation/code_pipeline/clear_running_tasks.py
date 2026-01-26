import ee
import json
from google.oauth2 import service_account
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(SCRIPT_DIR, 'my-dews-project-3d281b7bdeb3.json')

with open(KEY_FILE, 'r') as f:
    key_data = json.load(f)

credentials = service_account.Credentials.from_service_account_file(
    KEY_FILE,
    scopes=[
        'https://www.googleapis.com/auth/earthengine',
        'https://www.googleapis.com/auth/cloud-platform'
    ]
)

ee.Initialize(
    credentials=credentials,
    project=key_data['project_id'],
    opt_url='https://earthengine-highvolume.googleapis.com'
)

print("Fetching all tasks...")
tasks = ee.batch.Task.list()

ready = [t for t in tasks if t.state == 'READY']
running = [t for t in tasks if t.state == 'RUNNING']

print(f"\nFound {len(ready)} READY tasks")
print(f"Found {len(running)} RUNNING tasks")

if len(ready) + len(running) == 0:
    print("\n✅ No active tasks to cancel!")
else:
    print(f"\nCancelling {len(ready) + len(running)} tasks...")
    
    cancelled = 0
    for task in tasks:
        if task.state in ['READY', 'RUNNING']:
            try:
                task.cancel()
                cancelled += 1
                if cancelled % 10 == 0:
                    print(f"  Cancelled {cancelled} tasks...")
            except Exception as e:
                print(f"  Failed to cancel {task.id}: {e}")
    
    print(f"\n✅ Cancelled {cancelled} tasks total")
    print("Wait 30-60 seconds for cancellations to take effect.")
