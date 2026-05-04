import os
from huggingface_hub import HfApi

def fix_huggingface_space():
    print("=== Hugging Face Space Fixer ===")
    token = input("Please paste your Hugging Face Access Token (with Write permission): ").strip()
    
    if not token:
        print("Error: Token cannot be empty.")
        return

    api = HfApi(token=token)
    repo_id = "Akshanshsensei/PDF-Constrained-Conversational-Agent"
    repo_type = "space"

    files_to_delete = [
        "bm25_retriever.py", "chunker.py", "embedder.py", 
        "generator.py", "hybrid_retriever.py", "logger.py", 
        "memory.py", "parser.py", "reranker.py"
    ]

    print(f"\nConnecting to {repo_id}...")
    
    # 1. Delete the bad files
    for file_path in files_to_delete:
        try:
            print(f"Deleting {file_path} from the root directory...")
            api.delete_file(path_in_repo=file_path, repo_id=repo_id, repo_type=repo_type, commit_message=f"Delete misplaced {file_path}")
        except Exception as e:
            print(f"  (Skipped {file_path} - it might already be deleted or not exist)")

    print("\n✅ Bad files deleted successfully!")
    
    # 2. Upload the correct folders
    folders_to_upload = ["agent", "core", "utils"]
    
    for folder in folders_to_upload:
        if os.path.exists(folder):
            print(f"\nUploading correct {folder}/ folder...")
            api.upload_folder(
                folder_path=folder,
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload correct {folder} folder"
            )
        else:
            print(f"Warning: Could not find local folder {folder}/")

    print("\n🎉 All done! Your Hugging Face space should now build successfully.")
    print("Go check your space: https://huggingface.co/spaces/Akshanshsensei/PDF-Constrained-Conversational-Agent")

if __name__ == "__main__":
    fix_huggingface_space()
