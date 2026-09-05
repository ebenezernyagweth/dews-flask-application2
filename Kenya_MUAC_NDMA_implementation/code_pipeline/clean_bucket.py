from google.cloud import storage
from google.oauth2 import service_account

# ---- Config ----
KEY_FILE = 'my-dews-project-3d281b7bdeb3.json'  # adjust path if needed
BUCKET_NAME = "dews-muac-export"

# ---- Auth ----
credentials = service_account.Credentials.from_service_account_file(
    KEY_FILE,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
client = storage.Client(credentials=credentials, project='my-dews-project')
bucket = client.bucket(BUCKET_NAME)

# ---- Delete ALL blobs ----
blobs = list(bucket.list_blobs())  # no prefix = everything

if not blobs:
    print(f"[~] Bucket gs://{BUCKET_NAME} is already empty.")
else:
    print(f"[→] Found {len(blobs)} files in gs://{BUCKET_NAME}. Deleting...")
    for blob in blobs:
        try:
            blob.delete()
            print(f"    [✓] Deleted: {blob.name}")
        except Exception as e:
            print(f"    [✗] Failed to delete {blob.name}: {e}")

    print(f"\n[✓] Done. Bucket gs://{BUCKET_NAME} is now empty and ready for next run.")