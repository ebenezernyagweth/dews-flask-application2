#from google.cloud import storage
#from google.oauth2 import service_account
#import os
#
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#KEY_FILE = os.path.join(SCRIPT_DIR, 'my-dews-project-3d281b7bdeb3.json')
#
#credentials = service_account.Credentials.from_service_account_file(
#    KEY_FILE,
#    scopes=['https://www.googleapis.com/auth/cloud-platform']
#)
#
#client = storage.Client(credentials=credentials)
#bucket = client.bucket('dews-muac-exports')
#
# Test each permission individually
#permissions_to_test = [
#    'storage.objects.list',
#    'storage.objects.get',
#    'storage.objects.create',
#    'storage.objects.delete'
#]
#
#print("Testing permissions on bucket 'dews-muac-exports':")
#print("=" * 60)
#
#try:
#    result = bucket.test_iam_permissions(permissions_to_test)
#    print(f"\n✅ Service account HAS these permissions:")
#    for perm in result:
#        print(f"   ✓ {perm}")
#    
#    missing = set(permissions_to_test) - set(result)
#    if missing:
#        print(f"\n❌ Service account MISSING these permissions:")
#        for perm in missing:
#            print(f"   ✗ {perm}")
#    else:
#        print("\n🎉 All required permissions are granted!")
#        
#except Exception as e:
#    print(f"\n❌ Could not test permissions: {e}")


#import json
#
#with open('my-dews-project-3d281b7bdeb3.json', 'r') as f:
#    key_data = json.load(f)
#    print(f"Project ID: {key_data['project_id']}")
#    
#    # Project number might be in the file
#    if 'project_number' in key_data:
#        print(f"Project Number: {key_data['project_number']}")
#    
#    # Also check the private_key_id
#    print(f"\nFull service account: muacdews@my-dews-project.iam.gserviceaccount.com")
#


from google.cloud import iam_admin_v1
from google.oauth2 import service_account
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(SCRIPT_DIR, 'my-dews-project-3d281b7bdeb3.json')

credentials = service_account.Credentials.from_service_account_file(
    KEY_FILE,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

client = iam_admin_v1.IAMClient(credentials=credentials)

project_name = f"projects/my-dews-project"

print("Service accounts in your project:")
print("=" * 60)

try:
    request = iam_admin_v1.ListServiceAccountsRequest(name=project_name)
    page_result = client.list_service_accounts(request=request)
    
    for account in page_result:
        print(f"✓ {account.email}")
        
except Exception as e:
    print(f"Error: {e}")
