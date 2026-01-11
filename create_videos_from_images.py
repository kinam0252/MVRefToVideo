#!/usr/bin/env python3
"""
Convert images to videos with specific resolution and frame count.
- Resize images to 992x480 (add white padding on the right)
- Repeat frame 49 times
- Output as MP4 with 8 FPS
"""

import subprocess
from pathlib import Path
import sys

# Paths
INPUT_DIR = Path("/home/nas5/kinamkim/Repos/Fliption/DATA/processed/test/images")
OUTPUT_DIR = Path("/home/nas5/kinamkim/Repos/Fliption/DATA/processed/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Video settings
TARGET_WIDTH = 992
TARGET_HEIGHT = 480
TARGET_FPS = 8.0
TARGET_FRAMES = 49

def create_video_from_image(image_path, output_path):
    """
    Create video from image using ffmpeg.
    - Resize image to fit height 480, maintain aspect ratio
    - Add white padding on the right to make it 992x480
    - Loop frame 49 times
    - Output at 8 FPS
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # ffmpeg command:
    # 1. scale: scale to height 480, width auto (maintain aspect ratio)
    # 2. pad: add white padding on the right to make width 992
    #    pad=width:height:x:y:color
    #    x=(ow-iw)/2 centers horizontally, but we want right padding, so x=0
    # 3. loop: loop the input image
    # 4. frames: exactly 49 frames
    # 5. fps: set output fps to 8
    # 6. codec: use libx264
    
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-loop', '1',  # Loop input image
        '-i', str(image_path),  # Input image
        '-vf', f'scale=-1:{TARGET_HEIGHT},pad={TARGET_WIDTH}:{TARGET_HEIGHT}:0:0:white',  # Scale to height 480, pad to 992x480 with white on right
        '-frames:v', str(TARGET_FRAMES),  # Exactly 49 frames
        '-r', str(TARGET_FPS),  # Output frame rate
        '-vcodec', 'libx264',  # Video codec
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Created: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create {output_path.name}")
        print(f"Error: {e.stderr}")
        return False

def main():
    # Find all PNG images in input directory
    image_files = sorted(INPUT_DIR.glob("*.png"))
    
    if not image_files:
        print(f"❌ No PNG images found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Target FPS: {TARGET_FPS}, Frames: {TARGET_FRAMES}")
    print("-" * 50)
    
    success_count = 0
    for image_path in image_files:
        # Extract number from filename (e.g., "1.png" -> "1")
        image_name = image_path.stem
        output_path = OUTPUT_DIR / f"{image_name}.mp4"
        
        if create_video_from_image(image_path, output_path):
            success_count += 1
    
    print("-" * 50)
    print(f"✅ Successfully created {success_count}/{len(image_files)} videos")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

