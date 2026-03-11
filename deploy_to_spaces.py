"""
Deploy the Deepfake Detection app to Hugging Face Spaces.
Run: py deploy_to_spaces.py
"""

from huggingface_hub import HfApi, create_repo
import os

HF_USERNAME = "Purnachander-Konda"
SPACE_NAME = "deepfake-detection-swin"
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

# Files to upload to the Space
FILES = {
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",
}

# Minimal requirements for the Space (no GPU needed for inference)
SPACE_REQUIREMENTS = """gradio>=4.0.0
transformers>=4.36.0
torch>=2.0.0
Pillow>=9.0.0
"""

def main():
    token = input("Paste your Hugging Face token (from https://huggingface.co/settings/tokens): ").strip()

    api = HfApi(token=token)

    # Create the Space
    print(f"\n1. Creating Space: {REPO_ID} ...")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True,
        )
        print(f"   ✓ Space created!")
    except Exception as e:
        print(f"   Note: {e}")

    # Upload app.py
    print("\n2. Uploading app.py ...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    api.upload_file(
        path_or_fileobj=app_path,
        path_in_repo="app.py",
        repo_id=REPO_ID,
        repo_type="space",
    )
    print("   ✓ app.py uploaded!")

    # Upload a slim requirements.txt (only what the Space needs)
    print("\n3. Uploading requirements.txt ...")
    api.upload_file(
        path_or_fileobj=SPACE_REQUIREMENTS.encode(),
        path_in_repo="requirements.txt",
        repo_id=REPO_ID,
        repo_type="space",
    )
    print("   ✓ requirements.txt uploaded!")

    print("\n" + "=" * 60)
    print("🚀 DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print(f"\nYour live app: https://huggingface.co/spaces/{REPO_ID}")
    print(f"\nIt may take 2-3 minutes to build. Visit the link to check!")
    print("=" * 60)

if __name__ == "__main__":
    main()
