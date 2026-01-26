
import zipfile
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CODE_DIR = os.path.join(SCRIPT_DIR, "intermediary_datasets")  

zip_path = os.path.join(SCRIPT_DIR, "intermediary_datasets.zip")
extract_to = SCRIPT_DIR

def unzip_file(zip_path, extract_to):
    """
    Unzip the contents of a ZIP file into a specified directory.
    """
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extracted '{zip_path}' to '{extract_to}'")

unzip_file()

