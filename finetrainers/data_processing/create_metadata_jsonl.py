#!/usr/bin/env python3
"""
metadata.jsonl 생성 스크립트
captions.json을 읽어서 metadata.jsonl 형식으로 변환
"""
import json
import os
from pathlib import Path

# 경로 설정 (상대 경로)
# data_processing -> finetrainers
SCRIPT_DIR = Path(__file__).parent
FINETRAINERS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FINETRAINERS_ROOT.parent  # MVRefToVideo

CAPTIONS_JSON = FINETRAINERS_ROOT / "examples/training/sft/wan/mvref_lora/captions.json"
VIDEO_DIR = PROJECT_ROOT / "DATA/processed/videos"
OUTPUT_JSONL = VIDEO_DIR / "metadata.jsonl"

print("=" * 70)
print("Creating metadata.jsonl...")
print("=" * 70)
print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📂 Finetrainers root: {FINETRAINERS_ROOT}")
print(f"📄 Captions JSON: {CAPTIONS_JSON}")
print(f"📁 Output JSONL: {OUTPUT_JSONL}")
print("=" * 70)

# 파일 존재 확인
if not CAPTIONS_JSON.exists():
    print(f"❌ Error: Captions file not found!")
    print(f"   Expected: {CAPTIONS_JSON}")
    exit(1)

print(f"\n✅ Captions file found: {CAPTIONS_JSON}")

# captions 로드
print(f"\n[Step 1] Loading captions...")
with open(CAPTIONS_JSON, 'r', encoding='utf-8') as f:
    captions = json.load(f)
print(f"✅ Loaded {len(captions)} captions")

# metadata.jsonl 생성
print(f"\n[Step 2] Creating metadata.jsonl...")
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for video_id, caption in captions.items():
        metadata = {
            "file_name": f"{video_id}.mp4",
            "caption": caption
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"  ✅ {video_id}.mp4")

print(f"\n" + "=" * 70)
print(f"✅ Created: {OUTPUT_JSONL}")
print(f"   Total entries: {len(captions)}")
print("=" * 70)

# 샘플 확인
print("\n📊 Sample entry:")
with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    sample = json.loads(first_line)
    print(f"  file_name: {sample['file_name']}")
    caption_preview = sample['caption'][:100] + "..." if len(sample['caption']) > 100 else sample['caption']
    print(f"  caption: {caption_preview}")

print("\n✅ Done!")