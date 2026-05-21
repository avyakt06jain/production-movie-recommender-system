"""
Downloads model artifacts from Hugging Face Hub into ./artifacts/.

Only runs for files that are missing locally.  This is invoked:
  - During Docker build (RUN python scripts/download_artifacts.py)
  - On first cold-start if artifacts/ was not pre-populated

If HF credentials are not set (local dev with artifacts already present),
the script exits gracefully without error.
"""

import os
import sys

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


ARTIFACTS: list[str] = [
    "two_tower.pt",
    "lgbm_ranker.txt",
    "item_embeddings.npy",
    "item_ids.npy",
    "features.pkl",
]


def download_all(artifacts_dir: str = "artifacts") -> None:
    """Download any missing artifact files from Hugging Face Hub."""
    os.makedirs(artifacts_dir, exist_ok=True)

    missing = [
        fname for fname in ARTIFACTS
        if not os.path.exists(os.path.join(artifacts_dir, fname))
    ]

    if not missing:
        print("✅ All artifact files already present — nothing to download.")
        return

    repo_id = os.environ.get("HF_REPO_ID")
    token = os.environ.get("HF_TOKEN")

    if not repo_id:
        print(
            "⚠️  HF_REPO_ID environment variable not set.\n"
            "   Cannot download artifacts from Hugging Face Hub.\n"
            "   Make sure the following files exist locally:\n"
            + "\n".join(f"     artifacts/{f}" for f in missing)
        )
        return

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print(
            "⚠️  huggingface_hub package not installed. "
            "Install it with: pip install huggingface-hub"
        )
        return

    print(f"📦 Downloading {len(missing)} artifact(s) from {repo_id}…")

    for fname in missing:
        dest = os.path.join(artifacts_dir, fname)
        print(f"   ↓ {fname}…", end=" ", flush=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=artifacts_dir,
                token=token,
            )
            if os.path.exists(dest):
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
            else:
                print("✗ (file not found after download)")
        except Exception as exc:
            print(f"✗ ({exc})")

    print("Done.")


if __name__ == "__main__":
    download_all()
