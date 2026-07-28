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

print("Checking Earth Engine task status...")
tasks = ee.batch.Task.list()

ready = [t for t in tasks if t.state == 'READY']
running = [t for t in tasks if t.state == 'RUNNING']

print(f"READY: {len(ready)}")
print(f"RUNNING: {len(running)}")

total_active = len(ready) + len(running)

if total_active == 0:
    print("\n✅ Earth Engine is idle")
else:
    print(f"\n⚠️ Earth Engine is busy with {total_active} active task(s)")