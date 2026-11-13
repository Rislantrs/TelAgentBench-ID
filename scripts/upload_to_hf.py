"""Upload the TelAgentBench-ID evaluation dataset to Hugging Face Hub."""

import os
from huggingface_hub import HfApi

REPO_ID = "Rislantrs/TelAgentBench-ID"
DEFAULT_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "2_HuggingFace_Dataset_TelAgentBench-ID"),
    "dataset_telagentbench_indonesia_clean",
]


def resolve_dataset_dir():
    for candidate in DEFAULT_DIRS:
        path = os.path.abspath(candidate)
        if os.path.isdir(path):
            return path
    return None


def upload_dataset():
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
            commit_message="Upload TelAgentBench-ID evaluation benchmark across 23 Indonesian BSS APIs",
        )
        print(f"Upload complete: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as exc:
        print(f"Upload failed: {exc}")
        raise


if __name__ == "__main__":
    upload_dataset()
