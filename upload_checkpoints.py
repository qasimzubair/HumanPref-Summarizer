"""Upload checkpoints/ to a Hugging Face Hub repo so they can be linked in the README (do NOT commit them to GitHub).
Usage: HF_TOKEN=... python upload_checkpoints.py your-username/genai-a3-summarization
"""
import os
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path="checkpoints", repo_id=repo, path_in_repo="checkpoints")
print(f"Uploaded: https://huggingface.co/{repo}")
