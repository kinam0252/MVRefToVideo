#!/usr/bin/env python3
"""
TODO #2: MVREF 데이터셋 준비 (개선 버전)
- View 정보 명시
- Caption 길이 제한 제거
"""

import json
import os
from pathlib import Path

# ==================== 설정 ====================
RAW_DATASET_DIR = "/home/nas5/kinamkim/Repos/geonwoo/MVRefToVideo/DATA/raw_dataset/mv_images"
VIDEO_DIR = "/home/nas5/kinamkim/Repos/geonwoo/MVRefToVideo/DATA/processed/videos"
OUTPUT_DIR = "/home/nas5/kinamkim/Repos/geonwoo/MVRefToVideo/examples/training/sft/wan/mvref_lora"
ID_TOKEN = "MVREF"

# ==================== 함수 ====================

def read_caption_file(caption_path):
    """caption.txt 읽어서 파싱"""
    with open(caption_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # View별로 파싱
    views = {}
    current_view = None
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # View 헤더 감지
        if line.startswith('<') and line.endswith('>'):
            current_view = line.strip('<>')
            views[current_view] = []
        elif current_view:
            views[current_view].append(line)
    
    return views

def generate_caption_from_views(views):
    """View별 설명을 구조화된 caption으로 통합"""
    
    # View별로 명시적으로 caption 생성
    view_captions = []
    
    for view_name, view_desc in views.items():
        # View 설명 통합
        desc_text = ' '.join(view_desc)
        
        # View 이름 + 설명
        view_caption = f"[{view_name}] {desc_text}"
        view_captions.append(view_caption)
    
    # 전체 통합
    combined = ' '.join(view_captions)
    
    # ID_TOKEN 추가
    caption = f"{ID_TOKEN} A multi-view reference showing: {combined}"
    
    # ✅ 길이 제한 제거! (또는 2000자로 증가)
    # if len(caption) > 2000:
    #     caption = caption[:2000]
    
    return caption

def generate_simple_caption(video_id):
    """Caption 파일 없을 때 대체 caption"""
    return f"{ID_TOKEN} A multi-view reference video showing an object from six different angles: [Front View] front details, [Back View] back details, [Left View] left side details, [Right View] right side details, [Top View] top details, [Bottom View] bottom details, followed by a smooth rotating animation."

# ==================== Main ====================

print("=" * 70)
print("TODO #2: MVREF Dataset 준비 (개선 버전)")
print("=" * 70)

# 1. 비디오 파일 목록
print("\n[Step 1] 비디오 파일 확인...")
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
print(f"✅ Found {len(video_files)} videos")
print(f"   Range: {video_files[0]} ~ {video_files[-1]}")

# 2. Output 디렉토리 생성
print("\n[Step 2] Output 디렉토리 생성...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Created: {OUTPUT_DIR}")

# 3. Caption 생성
print("\n[Step 3] Caption 생성 (View 정보 포함)...")
captions = {}
missing_captions = []
stats = {
    'total_length': 0,
    'max_length': 0,
    'min_length': float('inf'),
    'with_views': 0
}

for video_file in video_files:
    video_id = video_file.replace('.mp4', '')
    caption_path = os.path.join(RAW_DATASET_DIR, video_id, 'caption.txt')
    
    if os.path.exists(caption_path):
        try:
            # caption.txt 읽기
            views = read_caption_file(caption_path)
            caption = generate_caption_from_views(views)
            captions[video_id] = caption
            
            # 통계
            cap_len = len(caption)
            stats['total_length'] += cap_len
            stats['max_length'] = max(stats['max_length'], cap_len)
            stats['min_length'] = min(stats['min_length'], cap_len)
            stats['with_views'] += len(views)
            
            print(f"   ✅ {video_id}: {cap_len} chars, {len(views)} views")
        except Exception as e:
            print(f"   ⚠️  {video_id}: Error - {e}")
            captions[video_id] = generate_simple_caption(video_id)
            missing_captions.append(video_id)
    else:
        print(f"   ⚠️  {video_id}: No caption.txt, using default")
        captions[video_id] = generate_simple_caption(video_id)
        missing_captions.append(video_id)

print(f"\n📊 Caption 통계:")
print(f"   Total: {len(captions)}")
print(f"   From caption.txt: {len(captions) - len(missing_captions)}")
print(f"   Default: {len(missing_captions)}")
print(f"   Average length: {stats['total_length'] // len(captions)} chars")
print(f"   Max length: {stats['max_length']} chars")
print(f"   Min length: {stats['min_length']} chars")
print(f"   Total views: {stats['with_views']}")

# 4. training.json 생성
print("\n[Step 4] training.json 생성...")
training_config = {
    "datasets": [
        {
            "data_root": VIDEO_DIR,
            "dataset_type": "video",
            "id_token": ID_TOKEN,
            "video_resolution_buckets": [[49, 480, 992]]
        }
    ]
}

training_path = os.path.join(OUTPUT_DIR, "training.json")
with open(training_path, "w") as f:
    json.dump(training_config, f, indent=2)
print(f"✅ Created: {training_path}")

# 5. captions.json 저장
print("\n[Step 5] captions.json 저장...")
captions_path = os.path.join(OUTPUT_DIR, "captions.json")
with open(captions_path, "w", encoding='utf-8') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)
print(f"✅ Created: {captions_path}")
print(f"   Total captions: {len(captions)}")

# 6. validation.json 생성
print("\n[Step 6] validation.json 생성...")

validation_prompts = [
    f"{ID_TOKEN} A multi-view reference video showing: [Front View] detailed front features, [Side View] side profile, [Bottom View] bottom details, [Top View] top surface, with smooth transitions between views",
    f"{ID_TOKEN} Multi-angle footage: [Front View] front-facing elements, [Back View] rear components, [Left View] left side, [Right View] right side, [Top View] overhead, [Bottom View] underside",
    f"{ID_TOKEN} A comprehensive reference: [Front View] primary face, [Side View] lateral view, [Bottom View] base details, with complete 360-degree coverage",
    f"{ID_TOKEN} Multi-view documentation: [Front View] frontal details, [Top View] top-down perspective, [Bottom View] bottom surface, showing all angles systematically"
]

validation_config = {
    "prompts": validation_prompts
}

validation_path = os.path.join(OUTPUT_DIR, "validation.json")
with open(validation_path, "w") as f:
    json.dump(validation_config, f, indent=2)
print(f"✅ Created: {validation_path}")
print(f"   Prompts: {len(validation_prompts)}")

# 7. 샘플 출력 (전체 caption 표시!)
print("\n[Step 7] Caption 샘플...")
print("=" * 70)
sample_ids = list(captions.keys())[:2]
for vid_id in sample_ids:
    print(f"\n📹 Video: {vid_id}.mp4")
    caption = captions[vid_id]
    print(f"   Length: {len(caption)} chars")
    # 처음 300자만 표시 (전체는 너무 김)
    if len(caption) > 300:
        print(f"   Caption: {caption[:300]}...")
    else:
        print(f"   Caption: {caption}")
    
    # View 카운트
    view_count = caption.count('[')
    print(f"   Views: {view_count}")
print("=" * 70)

# 8. 완료
print("\n" + "=" * 70)
print("✅ TODO #2 완료! (개선 버전)")
print("=" * 70)
print(f"\n생성된 파일:")
print(f"  1. {training_path}")
print(f"  2. {captions_path}")
print(f"  3. {validation_path}")

print(f"\n💡 개선 사항:")
print(f"  ✅ View 정보 명시 ([Front View], [Side View] 등)")
print(f"  ✅ Caption 길이 제한 제거 (전체 정보 보존)")
print(f"  ✅ 구조화된 caption 생성")

print(f"\n다음 단계:")
print(f"  1. captions.json 확인")
print(f"  2. train.sh 설정")
print(f"  3. 학습 시작!")