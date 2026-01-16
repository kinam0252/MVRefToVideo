#!/usr/bin/env python3
"""
Concatenate multiview images with video frames for successful samples only

Usage:
    python concat_mv_videos_success_only.py
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
    print(" imageio not found!")
    print()
    print("Please install it:")
    print("  pip install imageio imageio-ffmpeg --break-system-packages")
    print()
    exit(1)

# Configuration
SCRIPT_DIR = Path(__file__).parent
# Go up two levels: data_processing -> finetrainers -> MVRefToVideo
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_MV_DIR = PROJECT_ROOT / "_archive/data_preprocess/data/mv_images"
INPUT_VIDEO_DIR = PROJECT_ROOT / "_archive/data_preprocess/data/video_generated/videos"
OUTPUT_DIR = PROJECT_ROOT / "DATA/processed/videos_480x480"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WAN_TARGET_FPS = 8
WAN_TARGET_FRAMES = 49

# Failed samples to exclude (105 samples)
FAILED_SAMPLES = {
    33, 34, 47, 53, 76, 77, 83, 91, 94, 100,
    104, 123, 140, 155, 165, 170, 192, 195, 203, 233,
    248, 252, 260, 265, 286, 302, 320, 322, 325, 334,
    337, 352, 353, 354, 355, 367, 379, 383, 384, 389,
    396, 403, 425, 434, 437, 444, 453, 457, 469, 478,
    489, 492, 497, 508, 531, 536, 544, 553, 554, 557,
    563, 569, 574, 594, 608, 615, 627, 640, 648, 649,
    656, 662, 666, 679, 687, 693, 705, 714, 730, 731, 739,
    755, 774, 782, 783, 784, 802, 811, 836, 884, 888,
    891, 897, 901, 910, 913, 915, 947, 951, 961, 971,
    974, 975, 977, 988, 989
}


def resize_image_keep_aspect(image, target_height):
    """Resize image to target height while maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    
    # Ensure width is even (required by H.264)
    if new_width % 2 != 0:
        new_width += 1
    
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
    """Write video using imageio-ffmpeg (H.264 codec)"""
    # Convert BGR to RGB
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    # Check dimensions are even (H.264 requirement)
    h, w = frames_rgb[0].shape[:2]
    if w % 2 != 0 or h % 2 != 0:
        # Adjust to even dimensions
        new_w = w if w % 2 == 0 else w + 1
        new_h = h if h % 2 == 0 else h + 1
        frames_rgb = [cv2.resize(frame, (new_w, new_h)) for frame in frames_rgb]
    
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


def find_video_file(sample_id, video_dir):
    """
    Find video file for given sample_id.
    Videos are named like: {sample_id}_{description}.mp4
    """
    # Pattern 1: {sample_id}_{description}.ext
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Find files that start with "{sample_id}_"
        pattern = f"{sample_id}_*{ext}"
        matches = list(video_dir.glob(pattern))
        if matches:
            return matches[0]
    
    # Pattern 2: exact match {sample_id}.ext (fallback)
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        candidate = video_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    
    return None


def process_single_sample(sample_id, verbose=False):
    """Process a single sample"""
    
    # 1. Load multiview image
    mv_image_path = INPUT_MV_DIR / str(sample_id) / "final.png"
    if not mv_image_path.exists():
        if verbose:
            print(f"  Sample {sample_id}: final.png not found")
        return None
    
    mv_image = cv2.imread(str(mv_image_path))
    if mv_image is None:
        if verbose:
            print(f"  Sample {sample_id}: Failed to load multiview image")
        return None
    
    # Store original dimensions
    orig_h, orig_w = mv_image.shape[:2]
    
    if verbose:
        print(f"  MV image original: {orig_w}x{orig_h}")
    
    # 2. Find video file (with new pattern matching)
    video_path = find_video_file(sample_id, INPUT_VIDEO_DIR)
    
    if video_path is None:
        if verbose:
            print(f"  Sample {sample_id}: No video file found")
        return None
    
    # 3. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if verbose:
            print(f"  Sample {sample_id}: Failed to open video")
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
    
    if verbose:
        print(f"  Video resolution: {frame_width}x{frame_height}")
        print(f"  MV resized: {mv_width}x{frame_height}")
        print(f"  Concat size: {mv_width + frame_width}x{frame_height}")
    
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
            print(f"  Sample {sample_id}: No frames collected")
        return None
    
    # 7. Write video with imageio-ffmpeg (H.264)
    output_path = OUTPUT_DIR / f"{sample_id}.mp4"
    output_width = mv_width + frame_width
    output_height = frame_height
    
    try:
        write_video_with_imageio(concat_frames, output_path, WAN_TARGET_FPS)
    except Exception as e:
        if verbose:
            print(f"  Sample {sample_id}: Failed to write video: {e}")
        return None
    
    if verbose:
        print(f"  Sample {sample_id}: {len(concat_frames)} frames")
    
    return {
        'sample_id': str(sample_id),
        'output_path': str(output_path.relative_to(PROJECT_ROOT)),
        'video_file': video_path.name,
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


def get_success_sample_ids():
    """Get all successful sample IDs (excluding failed ones)"""
    sample_ids = []
    
    if not INPUT_MV_DIR.exists():
        print(f"Error: Directory not found: {INPUT_MV_DIR}")
        return sample_ids
    
    for sample_dir in INPUT_MV_DIR.iterdir():
        if sample_dir.is_dir():
            try:
                sample_id = int(sample_dir.name)
                # Only include if not in failed list
                if sample_id not in FAILED_SAMPLES:
                    sample_ids.append(sample_id)
            except ValueError:
                continue
    
    sample_ids.sort()
    return sample_ids


def main():
    print("=" * 70)
    print("MULTIVIEW + VIDEO CONCATENATION (Success Samples Only)")
    print("=" * 70)
    print()
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input MV: {INPUT_MV_DIR}")
    print(f"Input video: {INPUT_VIDEO_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Check directories exist
    if not INPUT_MV_DIR.exists():
        print(f" Error: MV directory not found: {INPUT_MV_DIR}")
        return
    
    if not INPUT_VIDEO_DIR.exists():
        print(f" Error: Video directory not found: {INPUT_VIDEO_DIR}")
        return
    
    # Check imageio-ffmpeg
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        print(f" imageio version: {imageio.__version__}")
        print(f" imageio-ffmpeg: OK")
    except Exception as e:
        print(f" Error: {e}")
        return
    
    print()
    
    # Get sample IDs
    sample_ids = get_success_sample_ids()
    
    if not sample_ids:
        print(f"No samples found in {INPUT_MV_DIR}")
        return
    
    total_samples = 994
    failed_count = len(FAILED_SAMPLES)
    expected_success = total_samples - failed_count
    
    print(f"Total samples: {total_samples}")
    print(f"Failed samples (excluded): {failed_count}")
    print(f"Expected success: {expected_success}")
    print(f"Found in directory: {len(sample_ids)}")
    print(f"Target: {WAN_TARGET_FRAMES} frames @ {WAN_TARGET_FPS} FPS")
    print()
    
    # Process samples
    results = []
    failed = []
    
    print("Creating concatenated videos...")
    for idx, sample_id in enumerate(tqdm(sample_ids, desc="Processing")):
        # First sample with verbose
        verbose = (idx == 0)
        result = process_single_sample(sample_id, verbose=verbose)
        
        if result is None:
            failed.append(sample_id)
        else:
            results.append(result)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {len(results)} samples")
    print(f"Failed during processing: {len(failed)} samples")
    print(f"Total excluded (preprocessing failed): {failed_count} samples")
    
    if failed:
        print(f"\nFailed during concat: {failed[:20]}")
        if len(failed) > 20:
            print(f"... and {len(failed) - 20} more")
    
    # Statistics
    if results:
        output_frames = [r['output_video']['num_frames'] for r in results]
        
        print(f"\nFrame statistics:")
        print(f"  Output: {min(output_frames)}-{max(output_frames)} frames")
        print(f"  Target: {WAN_TARGET_FRAMES} frames")
        
        # Resolution statistics
        resolutions = [f"{r['output_video']['resolution']['width']}x"
                      f"{r['output_video']['resolution']['height']}" 
                      for r in results]
        from collections import Counter
        resolution_counts = Counter(resolutions)
        most_common_res = resolution_counts.most_common(1)[0]
        
        print(f"\nOutput resolution:")
        print(f"  Most common: {most_common_res[0]} ({most_common_res[1]} samples)")
        
        # File size
        total_size = sum(
            (PROJECT_ROOT / r['output_path']).stat().st_size 
            for r in results
        ) / (1024**2)
        print(f"\nTotal output size: {total_size:.1f} MB")
        print(f"Average per video: {total_size/len(results):.2f} MB")
    
    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata = {
        'processing_info': {
            'total_processed': len(results),
            'failed_during_concat': len(failed),
            'excluded_preprocessing_failed': failed_count,
            'creation_date': datetime.now().isoformat(),
            'script_version': 'success_only_v1.1_fixed'
        },
        'wan_format': {
            'target_fps': WAN_TARGET_FPS,
            'target_frames': WAN_TARGET_FRAMES,
            'sampling_method': 'first_n_frames',
            'codec': 'H.264'
        },
        'excluded_samples': sorted(list(FAILED_SAMPLES)),
        'failed_concat': [str(sid) for sid in failed],
        'samples': results
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata: {metadata_path}")
    print("=" * 70)
    print(" COMPLETE!")
    print()
    print(f"Output: {OUTPUT_DIR}")
    print(f"Videos: {len(results)} files")
    print(f"Format: [multiview | video frame] @ {WAN_TARGET_FPS} FPS")


if __name__ == "__main__":
    main()