import requests
import os

url = "https://lznode.waondosecondary.xyz/web_display_image?imageID=12019245612f9d1e2b68881a1cfa9153"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

save_path = os.path.join(SCRIPT_DIR, "MUAC_Data_1.xlsx")

# save_path = r"C:\Users\Eodeyo\OneDrive - Farmers Choice Limited\Desktop\Muac_data.xlsx"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Download file
response = requests.get(url)

if response.status_code == 200:
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"File saved as: {save_path}")
else:
    print("Failed to download file. Status code:", response.status_code)