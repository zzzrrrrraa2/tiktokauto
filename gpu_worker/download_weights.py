#!/usr/bin/env python3
"""Pre-download ProPainter and PaddleOCR weights to /weights (Network Volume).
Run once to populate the RunPod Network Volume.
"""

import os
import sys
import subprocess

WEIGHTS_DIR = "/weights"
PROPAINTER_DIR = os.path.join(WEIGHTS_DIR, "propainter")
PADDLEOCR_DIR = os.path.join(WEIGHTS_DIR, "paddleocr")
TRANSNET_DIR = os.path.join(WEIGHTS_DIR, "transnetv2")

PROPAINTER_WEIGHTS = {
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
}


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} -> {dest}")
    subprocess.run(["wget", "-q", "--show-progress", "-O", dest, url], check=True)


def clone_propainter():
    propainter_repo = os.path.join(PROPAINTER_DIR, "repo")
    if not os.path.exists(propainter_repo):
        print("Cloning ProPainter repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/sczhou/ProPainter.git",
            propainter_repo
        ], check=True)

    # Create symlink structure for imports
    os.makedirs(os.path.join(PROPAINTER_DIR, "model", "modules"), exist_ok=True)
    os.makedirs(os.path.join(PROPAINTER_DIR, "core"), exist_ok=True)

    src = os.path.join(propainter_repo, "model")
    for item in os.listdir(src):
        dst = os.path.join(PROPAINTER_DIR, "model", item)
        if not os.path.exists(dst):
            if os.path.isdir(os.path.join(src, item)):
                os.symlink(os.path.join(src, item), dst, target_is_directory=True)
            else:
                os.symlink(os.path.join(src, item), dst)

    core_src = os.path.join(propainter_repo, "core")
    for item in os.listdir(core_src):
        dst = os.path.join(PROPAINTER_DIR, "core", item)
        if not os.path.exists(dst):
            os.symlink(os.path.join(core_src, item), dst)


def download_propainter_weights():
    os.makedirs(PROPAINTER_DIR, exist_ok=True)
    print("Downloading ProPainter weights...")
    for filename, url in PROPAINTER_WEIGHTS.items():
        download_file(url, os.path.join(PROPAINTER_DIR, filename))


def download_paddleocr():
    os.makedirs(PADDLEOCR_DIR, exist_ok=True)
    print("Downloading PaddleOCR models...")
    try:
        from paddleocr import PaddleOCR
        print("  Initializing PaddleOCR (models will be cached)...")
        ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
        print("  PaddleOCR ready.")
    except Exception as e:
        print(f"  PaddleOCR init warning: {e}")
        print("  Models will download on first handler invocation.")


def download_transnetv2_weights():
    os.makedirs(TRANSNET_DIR, exist_ok=True)
    weights_dir = os.path.join(TRANSNET_DIR, "transnetv2-weights")
    variables_dir = os.path.join(weights_dir, "variables")
    os.makedirs(variables_dir, exist_ok=True)

    print("Downloading TransNetV2 weights...")
    base_url = "https://huggingface.co/ToughStone/transnetv2-weights/resolve/main"

    download_file(
        f"{base_url}/saved_model.pb",
        os.path.join(weights_dir, "saved_model.pb")
    )
    download_file(
        f"{base_url}/variables/variables.index",
        os.path.join(variables_dir, "variables.index")
    )
    download_file(
        f"{base_url}/variables/variables.data-00000-of-00001",
        os.path.join(variables_dir, "variables.data-00000-of-00001")
    )
    print("  TransNetV2 weights ready.")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    clone_propainter()
    download_propainter_weights()

    try:
        download_paddleocr()
    except Exception as e:
        print(f"PaddleOCR optional setup: {e}")

    print("\nAll weights downloaded to /weights")
    print("Contents:")
    for root, dirs, files in os.walk(WEIGHTS_DIR):
        level = root.replace(WEIGHTS_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for file in files:
            if file.endswith(".pth") or file.endswith(".safetensors"):
                size_mb = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
                print(f"{indent}  {file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
