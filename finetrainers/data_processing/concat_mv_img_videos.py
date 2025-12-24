#!/usr/bin/env python3
"""
Concatenate multiview images with video frames using imageio-ffmpeg
SIMPLIFIED VERSION - assumes standardized input

- MV images are already 512 width (vertical concat of 6 views)
- Direct horizontal concatenation with video frames
- Generates H.264 videos for compatibility

Installation:
    pip install imageio imageio-ffmpeg --break-system-packages

Usage:
    python concat_mv_img_videos_standardized.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

try:
    import imageio
    import imageio_ffmpeg as ffmpeg
except ImportError:
    print("❌ imageio not found!")
    print()
    print("Please install it:")
    print("  pip install imageio imageio-ffmpeg --break-system-packages")
    print()
    exit(1)

# Configuration (상대 경로)
# data_processing -> finetrainers -> MVRefToVideo
SCRIPT_DIR = Path(__file__).parent
FINETRAINERS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FINETRAINERS_ROOT.parent

INPUT_MV_DIR = PROJECT_ROOT / "DATA/raw_dataset/mv_images"
INPUT_VIDEO_DIR = PROJECT_ROOT / "DATA/raw_dataset/videos"
OUTPUT_DIR = PROJECT_ROOT / "DATA/processed/videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WAN_TARGET_FPS = 8
WAN_TARGET_FRAMES = 49


def resize_image_keep_aspect(image, target_height):
    """
    Resize image to target height while maintaining aspect ratio
    """
    h, w = image.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    
    resized = cv2.resize(image, (new_width, target_height), 
                        interpolation=cv2.INTER_LANCZOS4)
    return resized


def get_first_n_frames(total_frames, target_frames):
    """Get indices for first N frames"""
    if total_frames <= target_frames:
        return np.arange(total_frames)
    else:
        return np.arange(target_frames)


def write_video_with_imageio(frames, output_path, fps):
    """
    Write video using imageio-ffmpeg (H.264 codec)
    """
    # Convert BGR to RGB (OpenCV uses BGR, imageio uses RGB)
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    # Write video with H.264 codec
    imageio.mimwrite(
        str(output_path),
        frames_rgb,
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
        macro_block_size=1
    )


def process_single_sample(sample_id, verbose=False):
    """
    Process a single sample
    
    Args:
        sample_id: Sample ID (int or str)
        verbose: Whether to print detailed logs
    
    Returns:
        dict: Metadata of processed sample, or None if failed
    """
    
    # 1. Load multiview image
    mv_image_path = INPUT_MV_DIR / str(sample_id) / "final.png"
    if not mv_image_path.exists():
        if verbose:
            print(f"⚠️  Sample {sample_id}: final.png not found")
        return None
    
    mv_image = cv2.imread(str(mv_image_path))
    if mv_image is None:
        if verbose:
            print(f"❌ Sample {sample_id}: Failed to load multiview image")
        return None
    
    # Store original dimensions
    orig_h, orig_w = mv_image.shape[:2]
    
    # 2. Find video file
    video_path = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        candidate = INPUT_VIDEO_DIR / f"{sample_id}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    
    if video_path is None:
        if verbose:
            print(f"⚠️  Sample {sample_id}: No video file found")
        return None
    
    # 3. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if verbose:
            print(f"❌ Sample {sample_id}: Failed to open video")
        return None
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 4. Get first N frames
    frame_indices = get_first_n_frames(total_frames, WAN_TARGET_FRAMES)
    
    # 5. Resize multiview image to match video height
    mv_image_resized = resize_image_keep_aspect(mv_image, frame_height)
    mv_width = mv_image_resized.shape[1]
    
    # 6. Collect concat frames
    concat_frames = []
    
    for frame_idx in range(len(frame_indices)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Horizontal concatenation: [multiview | video frame]
        concat_frame = np.hstack([mv_image_resized, frame])
        concat_frames.append(concat_frame)
    
    cap.release()
    
    if len(concat_frames) == 0:
        if verbose:
            print(f"❌ Sample {sample_id}: No frames collected")
        return None
    
    # 7. Write video with imageio-ffmpeg (H.264)
    output_path = OUTPUT_DIR / f"{sample_id}.mp4"
    output_width = mv_width + frame_width
    output_height = frame_height
    
    try:
        write_video_with_imageio(concat_frames, output_path, WAN_TARGET_FPS)
    except Exception as e:
        if verbose:
            print(f"❌ Sample {sample_id}: Failed to write video: {e}")
        return None
    
    if verbose:
        print(f"✅ Sample {sample_id}: {len(concat_frames)} frames → H.264")
    
    return {
        'sample_id': str(sample_id),
        'output_path': str(output_path.relative_to(PROJECT_ROOT)),
        'multiview': {
            'width': orig_w,
            'height': orig_h
        },
        'original_video': {
            'total_frames': total_frames,
            'fps': original_fps,
            'resolution': [frame_width, frame_height]
        },
        'output_video': {
            'num_frames': len(concat_frames),
            'fps': WAN_TARGET_FPS,
            'resolution': {
                'width': output_width,
                'height': output_height
            },
            'codec': 'H.264'
        },
        'components': {
            'multiview_width': mv_width,
            'video_width': frame_width,
            'height': frame_height
        }
    }


def get_all_sample_ids():
    """Get all sample IDs from mv_images directory"""
    sample_ids = []
    
    if not INPUT_MV_DIR.exists():
        print(f"❌ Error: Directory not found: {INPUT_MV_DIR}")
        return sample_ids
    
    for sample_dir in INPUT_MV_DIR.iterdir():
        if sample_dir.is_dir():
            dir_name = sample_dir.name
            try:
                sample_id = int(dir_name)
            except ValueError:
                sample_id = dir_name
            sample_ids.append(sample_id)
    
    sample_ids.sort(key=lambda x: (isinstance(x, str), x))
    return sample_ids


def main():
    print("="*70)
    print("📹 Multiview + Video Concatenation (Simplified)")
    print("   H.264 codec for maximum compatibility")
    print("="*70)
    print()
    
    print(f"📂 Project root: {PROJECT_ROOT}")
    print(f"📂 Input MV: {INPUT_MV_DIR.relative_to(PROJECT_ROOT)}")
    print(f"📂 Input video: {INPUT_VIDEO_DIR.relative_to(PROJECT_ROOT)}")
    print(f"📂 Output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print()
    
    # Check imageio-ffmpeg
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        print(f"✅ imageio version: {imageio.__version__}")
        print(f"✅ imageio-ffmpeg found")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print()
    
    # Get sample IDs
    sample_ids = get_all_sample_ids()
    
    if not sample_ids:
        print(f"❌ No samples found in {INPUT_MV_DIR}")
        return
    
    print(f"📊 Found {len(sample_ids)} samples")
    print(f"🎯 Target: {WAN_TARGET_FRAMES} frames @ {WAN_TARGET_FPS} FPS (H.264)")
    print()
    
    # Process samples
    results = []
    failed = []
    
    print("🔄 Creating H.264 videos...")
    for sample_id in tqdm(sample_ids, desc="Processing"):
        result = process_single_sample(sample_id, verbose=False)
        
        if result is None:
            failed.append(sample_id)
        else:
            results.append(result)
    
    print()
    print("="*70)
    print("📊 Summary")
    print("="*70)
    print(f"✅ Successfully processed: {len(results)} samples")
    print(f"❌ Failed: {len(failed)} samples")
    
    if failed:
        print(f"\n⚠️  Failed sample IDs: {failed}")
    
    # Statistics
    if results:
        output_frames = [r['output_video']['num_frames'] for r in results]
        
        print(f"\n📊 Frame statistics:")
        print(f"   Output: {min(output_frames)}-{max(output_frames)} frames")
        print(f"   Target: {WAN_TARGET_FRAMES} frames")
        
        # Resolution statistics
        resolutions = [f"{r['output_video']['resolution']['width']}x"
                      f"{r['output_video']['resolution']['height']}" 
                      for r in results]
        from collections import Counter
        resolution_counts = Counter(resolutions)
        most_common_res = resolution_counts.most_common(1)[0]
        
        print(f"\n📏 Output resolution:")
        print(f"   Most common: {most_common_res[0]} ({most_common_res[1]} samples)")
        
        # File size
        total_size = sum(
            (PROJECT_ROOT / r['output_path']).stat().st_size 
            for r in results
        ) / (1024**2)
        print(f"\n💾 Total output size: {total_size:.1f} MB")
    
    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata = {
        'processing_info': {
            'total_samples': len(results),
            'failed_samples': len(failed),
            'creation_date': datetime.now().isoformat(),
            'script_version': '6.0_simplified',
            'method': 'direct_concatenation'
        },
        'wan_format': {
            'target_fps': WAN_TARGET_FPS,
            'target_frames': WAN_TARGET_FRAMES,
            'sampling_method': 'first_n_frames',
            'codec': 'H.264'
        },
        'failed_ids': [str(sid) for sid in failed],
        'samples': results
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    print("="*70)
    print("✅ Processing complete!")
    print()
    print("🎬 All videos are now in H.264 format!")
    print(f"   - {WAN_TARGET_FRAMES} frames @ {WAN_TARGET_FPS} FPS")
    print(f"   - Each frame: [multiview | video frame]")
    print(f"   - Playable everywhere!")


if __name__ == "__main__":
    main()
