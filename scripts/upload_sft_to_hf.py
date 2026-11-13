"""Upload the TelkomNusa-SFT-ID training and validation corpus to Hugging Face Hub."""

import os
from huggingface_hub import HfApi

REPO_ID = "Rislantrs/TelkomNusa-SFT-ID"
DEFAULT_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "3_HuggingFace_Dataset_TelkomNusa-SFT-ID"),
    "dataset_sft_telko_indonesia_terbaru",
]


def resolve_dataset_dir():
    for candidate in DEFAULT_DIRS:
        path = os.path.abspath(candidate)
        if os.path.isdir(path):
            return path
    return None


def upload_sft_dataset():
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    local_folder = resolve_dataset_dir()
    if local_folder is None:
        raise FileNotFoundError(
            f"Dataset folder not found in expected paths: {DEFAULT_DIRS}"
        )

    print(f"Uploading {local_folder} to {REPO_ID}...")
    api = HfApi(token=hf_token)

    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=local_folder,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Upload TelkomNusa-SFT-ID corpus (3,129 train / 381 val dialogues across 23 BSS tools)",
        )
        print(f"Upload complete: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as exc:
        print(f"Upload failed: {exc}")
        raise


if __name__ == "__main__":
    upload_sft_dataset()
