#!/usr/bin/env python3
"""
Analyze images to identify what objects they contain.
"""

from pathlib import Path
from PIL import Image
import sys

INPUT_DIR = Path("/home/nas5/kinamkim/Repos/Fliption/DATA/processed/test/images")

def analyze_image(image_path):
    """Analyze an image and return basic info"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        mode = img.mode
        return {
            'path': image_path.name,
            'size': f"{width}x{height}",
            'mode': mode,
            'format': img.format
        }
    except Exception as e:
        return {
            'path': image_path.name,
            'error': str(e)
        }

def main():
    image_files = sorted(INPUT_DIR.glob("*.png"))
    
    if not image_files:
        print(f"❌ No PNG images found in {INPUT_DIR}")
        sys.exit(1)
    
    print("Image Analysis:")
    print("=" * 60)
    
    for img_path in image_files:
        info = analyze_image(img_path)
        print(f"\n{info['path']}:")
        if 'error' in info:
            print(f"  Error: {info['error']}")
        else:
            print(f"  Size: {info['size']}")
            print(f"  Mode: {info['mode']}")
            print(f"  Format: {info['format']}")
    
    # Also check if we can get category info from previous context
    # Based on previous conversation, these came from fliption_1228_v2 categories
    print("\n" + "=" * 60)
    print("Category mapping (from previous context):")
    categories = {
        '1.png': '가방 (Bag)',
        '2.png': '모자 (Hat)',
        '3.png': '안경 (Glasses)',
        '4.png': '운동화 (Sneakers)',
        '5.png': '음료 (Beverage)',
        '6.png': '의자 (Chair)',
        '7.png': '카메라 (Camera)',
        '8.png': '헤드셋 (Headset)',
        '9.png': '화장품 (Cosmetics)',
        '10.png': '휴대폰 (Mobile Phone)'
    }
    
    for filename, category in categories.items():
        if (INPUT_DIR / filename).exists():
            print(f"  {filename}: {category}")

if __name__ == "__main__":
    main()


