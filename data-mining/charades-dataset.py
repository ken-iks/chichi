"""
This script can be executed in a google colab notebook.

The Charades-STA dataset contains 9894 videos, and was used for this paper: https://arxiv.org/pdf/2210.12977

Link to the actual colab notebook for the last time the script was run (11/23) here: https://colab.research.google.com/drive/1uj3-hTL-YLrw_trdOjr9w_ufd_qlY6YP#scrollTo=QPSxZzQZMqt6
"""

# ==============================================================================
# Charades-STA Dataset Downloader & Uploader
# ==============================================================================
# Description:
#   1. Downloads Charades videos (480p) from Hugging Face Mirror (lmms-lab).
#   2. Downloads Charades-STA annotations from the official TALL GitHub.
#   3. Extracts video parts securely (handling them as independent zips).
#   4. Uploads the complete, labelled dataset to Google Cloud Storage.
# ==============================================================================

import os
import subprocess
import sys
from google.colab import auth

# --- CONFIGURATION ---
# Target Bucket Folder (Where the data will live)
BUCKET_NAME = "gs://vidz-v0/chichi-v0"


# ---------------------

def run_command(command, check=True):
    """Helper to run shell commands with logging."""
    try:
        subprocess.run(command, check=check, shell=False)
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing {command}: {e}")
        raise


def main():
    print("=== STEP 1: AUTHENTICATION ===")
    auth.authenticate_user()
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    # Quietly set the project
    subprocess.run(f"gcloud config set project {project_id}", shell=True)
    print(f"Authenticated project: {project_id}")

    print("\n=== STEP 2: INSTALLING DEPENDENCIES ===")
    # 1. Install 7zip (System tool)
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(["apt-get", "install", "-y", "p7zip-full"], check=True)

    # 2. Install HuggingFace Hub (Python tool)
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "-q"], check=True)

    # Import after install
    from huggingface_hub import hf_hub_download

    print("\n=== STEP 3: DOWNLOADING DATA ===")

    # A. Download Annotations (Text Files)
    print("Downloading Annotations...")
    os.makedirs("charades_dataset", exist_ok=True)
    subprocess.run(["wget", "-q", "-O", "charades_dataset/charades_sta_train.txt",
                    "https://raw.githubusercontent.com/jiyanggao/TALL/master/data/charades_sta_train.txt"])
    subprocess.run(["wget", "-q", "-O", "charades_dataset/charades_sta_test.txt",
                    "https://raw.githubusercontent.com/jiyanggao/TALL/master/data/charades_sta_test.txt"])

    # B. Download Videos (Hugging Face)
    print("Downloading Video Parts (Total ~16GB)...")
    video_parts = [
        "Charades_v1_480_part_1.zip",
        "Charades_v1_480_part_2.zip",
        "Charades_v1_480_part_3.zip",
        "Charades_v1_480_part_4.zip"
    ]

    for filename in video_parts:
        print(f"  - Fetching {filename}...")
        hf_hub_download(
            repo_id="lmms-lab/charades_sta",
            filename=filename,
            repo_type="dataset",
            local_dir=".",
            local_dir_use_symlinks=False,
            force_download=False  # Skip if already exists
        )

    print("\n=== STEP 4: EXTRACTING VIDEOS ===")
    # Note: These files are independent zips, not split volumes.
    # We extract them one by one into the same folder.
    for filename in video_parts:
        if os.path.exists(filename):
            print(f"  - Extracting {filename}...")
            # -y: Assume Yes to overwrite (merges folders)
            # -o: Output directory
            subprocess.run(["7z", "x", filename, "-ocharades_dataset/", "-y"], stdout=subprocess.DEVNULL)
        else:
            print(f"  - CRITICAL WARNING: {filename} missing!")

    # Verification
    video_folder = "charades_dataset/Charades_v1_480"
    if os.path.exists(video_folder):
        count = len(os.listdir(video_folder))
        print(f"\nExtraction Complete. Found {count} videos.")

        if count > 9000:
            print("\n=== STEP 5: UPLOADING TO GCS ===")
            print(f"Target: {BUCKET_NAME}")
            print("Starting upload (this may take 10-20 mins)...")
            # -m for multi-threaded, -r for recursive
            os.system(f"gsutil -m rsync -r charades_dataset {BUCKET_NAME}")
            print("\nSUCCESS: Dataset setup complete.")
        else:
            print(f"ERROR: Expected ~9800 videos, but found {count}. Check downloads.")
    else:
        print("ERROR: Extraction folder not found.")


if __name__ == "__main__":
    main()