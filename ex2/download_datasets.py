import os
import requests
import zipfile
from pathlib import Path

def download_and_extract(url, dataset_name, parent_dir):
    # Create the dataset folder
    dataset_dir = os.path.join(parent_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Get the file name from the URL
    file_name = url.split('/')[-1]

    # Define the full path for the ZIP file
    zip_path = os.path.join(dataset_dir, file_name)

    print(f"Downloading {dataset_name}...")
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {file_name}.")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")
        return

    # Extract the ZIP file
    print(f"Extracting {file_name} into {dataset_name} folder...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print(f"Extracted {file_name}.")

    # Remove the ZIP file after extraction
    os.remove(zip_path)
    print(f"Removed {file_name}.")

def main():
    # URLs for the datasets
    datasets = {
        "bike_sharing": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        "online_news_popularity": "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
    }

    # Define the target directory
    target_dir = os.path.join(Path(__file__).resolve().parent.parent, "data")

    # Download and extract each dataset
    for dataset_name, url in datasets.items():
        print(f"Processing {dataset_name}...")
        download_and_extract(url, dataset_name, target_dir)
    print("All datasets have been downloaded and extracted.")

if __name__ == "__main__":
    main()