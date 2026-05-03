import os
from huggingface_hub import HfApi

def upload_app():
    print("=== Hugging Face Full Sync Uploader ===")
    token = input("Please paste your Hugging Face Access Token (with Write permission): ").strip()
    
    if not token:
        print("Error: Token cannot be empty.")
        return

    api = HfApi(token=token)
    repo_id = "Akshanshsensei/PDF-Constrained-Conversational-Agent"
    repo_type = "space"

    # All files that need to be kept in sync with the Space
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "README.md",
        "config.py",
    ]

    print(f"\nUploading {len(files_to_upload)} files to Hugging Face Space...")
    success_count = 0
    for file_path in files_to_upload:
        if not os.path.exists(file_path):
            print(f"  ⚠️  Skipping {file_path} (not found locally)")
            continue
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Sync {file_path}"
            )
            print(f"  ✅ {file_path} uploaded.")
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error uploading {file_path}: {e}")

    if success_count > 0:
        print(f"\n✅ Done! {success_count}/{len(files_to_upload)} files uploaded.")
        print("Your Space is now rebuilding with the latest changes!")
    else:
        print("\n❌ No files were uploaded. Check your token and try again.")

if __name__ == "__main__":
    upload_app()
