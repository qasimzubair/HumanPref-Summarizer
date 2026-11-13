"""Download checkpoints (and optionally results) from the Hugging Face Hub into this repo.

Usage:  python download_checkpoints.py [user/repo] [--with-results]
Creates checkpoints/{sft,reward,ppo,dpo}/ next to this file. If the repo is private, run `huggingface-cli login` first.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

args = [a for a in sys.argv[1:] if not a.startswith("--")]
repo = args[0] if args else "qasimzubair/tldr-align"
patterns = ["checkpoints/*"] + (["results/*"] if "--with-results" in sys.argv else [])
snapshot_download(repo_id=repo, allow_patterns=patterns, local_dir=str(Path(__file__).resolve().parent))
print("Done. Contents of checkpoints/:", sorted(p.name for p in (Path(__file__).resolve().parent / "checkpoints").iterdir()))
