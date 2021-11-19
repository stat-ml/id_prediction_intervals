import os
import requests
import zipfile


def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading [{url}] to [{path}]...")
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
    else:
        print(f"File [{path}] already exists, skipping download.")


def download_file_to_dir(url, base_dir, download_file_name):
    download_path = os.path.join(base_dir, download_file_name)
    download_file(url, download_path)
    return download_path


def download_and_extract_zip(url, base_dir, download_file_name, extract_file_name):
    download_path = download_file_to_dir(url, base_dir, download_file_name)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)
    file_path = os.path.join(base_dir, extract_file_name)
    return file_path
