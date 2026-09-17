"""
Download pre-trained open-source YOLO models for maritime safety detection.

Models used:
  fire_smoke  — TommyNgx/YOLOv10-Fire-and-Smoke-Detection (HuggingFace)
  smoking     — Enos-123/smoking-detection (HuggingFace, YOLOv11)
  phone       — IndUSV/yolov8n-mobile-phone (HuggingFace)
  ppe         — ayushgupta7777/safetyvision-yolov8 (HuggingFace)
  person      — ultralytics YOLOv8n (built-in, COCO person class)
  fatigue     — user-provided fatigue/drowsiness detection model (YOLOv8)

Usage:
    cd ai_service
    python tools/download_models.py
    python tools/download_models.py --model fire_smoke   # single model
    python tools/download_models.py --list               # show status only

    # 国内加速（中国镜像）:
    set HF_ENDPOINT=https://hf-mirror.com && python tools/download_models.py
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# HuggingFace endpoint — override with HF_ENDPOINT env var for China mirror
# Usage: set HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

MODELS_DIR = Path(__file__).parent.parent / "models"

# Each entry: (filename, huggingface_repo, filename_in_repo)
MODEL_REGISTRY = {
    "fire_smoke": {
        "filename": "fire_smoke_yolov10.pt",
        "hf_repo": "TommyNgx/YOLOv10-Fire-and-Smoke-Detection",
        "hf_file": "best.pt",
        "description": "Fire & smoke detection (YOLOv10), labels: fire, smoke",
        "classes": ["fire", "smoke"],
    },
    "smoking": {
        "filename": "smoking_yolov11.pt",
        "hf_repo": "Enos-123/smoking-detection",
        "hf_file": "best.pt",
        "description": "Smoking / cigarette detection (YOLOv11-M), labels: smoking",
        "classes": ["smoking", "cigarette"],
    },
    "phone": {
        "filename": "phone_yolov8.pt",
        "hf_repo": "IndUSV/yolov8n-mobile-phone",
        "hf_file": "best.pt",
        "description": "Mobile phone detection (YOLOv8n), labels: phone",
        "classes": ["mobile_phone"],
    },
    "ppe": {
        "filename": "ppe_yolov8.pt",
        "hf_repo": "ayushgupta7777/safetyvision-yolov8",
        "hf_file": "best.pt",
        "description": "PPE detection (YOLOv8): helmet, vest, no_helmet, no_vest",
        "classes": ["helmet", "no_helmet", "vest", "no_vest", "person"],
    },
    "person": {
        "filename": "yolov8n.pt",
        "hf_repo": None,  # downloaded directly via ultralytics
        "hf_file": None,
        "description": "Person detection (YOLOv8n COCO), used for absence detection",
        "classes": ["person"],
    },
    "fatigue": {
        "filename": "fatigue_yolov8.pt",
        "hf_repo": None,  # user must provide this model manually
        "hf_file": None,
        "description": "Fatigue/drowsiness detection (YOLOv8), user-provided model",
        "classes": ["drowsy", "sleeping", "fatigue"],
    },
}


def _show_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        print(f"\r  [{'█' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)
    else:
        print(f"\r  Downloaded {downloaded / 1048576:.1f} MB", end="", flush=True)


def download_from_huggingface(repo: str, hf_file: str, dest: Path) -> bool:
    """Download from HuggingFace via direct HTTP — no extra dependencies needed."""
    url = f"{HF_ENDPOINT}/{repo}/resolve/main/{hf_file}"
    print(f"  Downloading: {url}")
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_show_progress)
        print()
        tmp.rename(dest)
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def download_via_ultralytics(model_name: str, dest: Path) -> bool:
    """Download a standard Ultralytics model (e.g. yolov8n.pt)."""
    try:
        from ultralytics import YOLO
        print(f"  Downloading {model_name} via Ultralytics ...")
        model = YOLO(model_name)  # auto-downloads to ultralytics cache
        # Copy from ultralytics cache to our models dir
        cache_path = Path.home() / ".ultralytics" / "assets" / model_name
        if not cache_path.exists():
            # Try alternate cache locations
            import ultralytics
            alt = Path(ultralytics.__file__).parent / "assets" / model_name
            if alt.exists():
                cache_path = alt
        if cache_path.exists():
            import shutil
            shutil.copy(cache_path, dest)
        else:
            # ultralytics may have written it to cwd
            cwd_path = Path.cwd() / model_name
            if cwd_path.exists():
                import shutil
                shutil.move(str(cwd_path), dest)
            else:
                # Model is already accessible by name from YOLO(); save a placeholder
                model.save(str(dest))
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  Ultralytics download failed: {e}")
        return False


def show_status():
    """Print current download status for all models."""
    print(f"\nModel directory: {MODELS_DIR}\n")
    print(f"{'Name':<12} {'File':<28} {'Status':<10} Description")
    print("-" * 90)
    for name, info in MODEL_REGISTRY.items():
        dest = MODELS_DIR / info["filename"]
        status = "✓ found" if dest.exists() else "✗ missing"
        print(f"{name:<12} {info['filename']:<28} {status:<10} {info['description']}")
    print()


def download_model(name: str) -> bool:
    """Download a single model by registry name."""
    if name not in MODEL_REGISTRY:
        print(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
        return False

    info = MODEL_REGISTRY[name]
    dest = MODELS_DIR / info["filename"]

    if dest.exists():
        print(f"[{name}] Already exists: {dest}")
        return True

    print(f"\n[{name}] {info['description']}")

    if info["hf_repo"] is None:
        # Standard ultralytics model
        ok = download_via_ultralytics(info["filename"], dest)
    else:
        ok = download_from_huggingface(info["hf_repo"], info["hf_file"], dest)
        if not ok:
            print(f"  Retrying via alternative URL ...")
            ok = _try_fallback(name, dest)

    return ok


def _try_fallback(name: str, dest: Path) -> bool:
    """Fallback: print manual download instructions."""
    info = MODEL_REGISTRY[name]
    if info["hf_repo"]:
        url = f"https://huggingface.co/{info['hf_repo']}/resolve/main/{info['hf_file']}"
        print(f"\n  Manual download: {url}")
        print(f"  Save as: {dest}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for maritime-vision AI service")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Download a single model")
    parser.add_argument("--list", action="store_true", help="Show download status only")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        show_status()
        return

    show_status()

    targets = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    results = {}
    for name in targets:
        results[name] = download_model(name)

    print("\n--- Summary ---")
    for name, ok in results.items():
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")

    missing = [n for n, ok in results.items() if not ok]
    if missing:
        print(f"\nSome models missing: {missing}")
        print("The AI service will skip missing models at startup and run with available ones.")
        sys.exit(1)
    else:
        print("\nAll models ready.")


if __name__ == "__main__":
    main()
