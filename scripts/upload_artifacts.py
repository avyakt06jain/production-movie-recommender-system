import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

def main():
    load_dotenv()
    
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not repo_id:
        logger.error("HF_TOKEN and HF_REPO_ID must be set in .env")
        return
        
    api = HfApi(token=hf_token)
    
    # Check if repo exists, create if it doesn't
    try:
        api.model_info(repo_id)
        logger.info(f"Repository {repo_id} exists.")
    except Exception:
        logger.info(f"Creating private repository {repo_id}...")
        api.create_repo(repo_id=repo_id, private=True, repo_type="model")
        
    artifacts_dir = Path("artifacts")
    files_to_upload = [
        "features.pkl",
        "item_embeddings.npy",
        "item_ids.npy",
        "lgbm_ranker.txt",
        "two_tower.pt"
    ]
    
    for filename in files_to_upload:
        file_path = artifacts_dir / filename
        if not file_path.exists():
            logger.warning(f"File {file_path} not found, skipping.")
            continue
            
        logger.info(f"Uploading {filename} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"Successfully uploaded {filename}")
        
    logger.info("All artifacts uploaded successfully!")

if __name__ == "__main__":
    main()
