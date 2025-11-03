"""
Image Resizer for Stable Diffusion Training
Resizes all images in raw/ folder to 512x512 or 768x768
Saves to processed/ folder
"""

from PIL import Image
import os
from pathlib import Path

# Configuration
RAW_FOLDER = "raw"
PROCESSED_FOLDER = "processed"
TARGET_SIZE = 512  # Change to 768 if you want higher resolution

# Supported formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}

def resize_image(input_path, output_path, size=512):
    """
    Resize image to square dimensions using center crop
    """
    try:
        # Open image
        img = Image.open(input_path)
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get dimensions
        width, height = img.size
        
        # Crop to square (center crop)
        if width != height:
            # Find the shorter side
            min_dim = min(width, height)
            
            # Calculate crop box (center)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            
            img = img.crop((left, top, right, bottom))
        
        # Resize to target size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Save as high-quality JPG
        img.save(output_path, 'JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")
        return False

def main():
    # Create output folder if it doesn't exist
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Get all images from raw folder
    raw_path = Path(RAW_FOLDER)
    
    if not raw_path.exists():
        print(f"❌ Error: {RAW_FOLDER} folder not found!")
        print("Please create the folder and add your images.")
        return
    
    # Find all image files
    image_files = [
        f for f in raw_path.iterdir() 
        if f.suffix.lower() in SUPPORTED_FORMATS
    ]
    
    if not image_files:
        print(f"❌ No images found in {RAW_FOLDER}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"🖼️  Found {len(image_files)} images")
    print(f"📏 Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"📂 Output folder: {PROCESSED_FOLDER}")
    print("-" * 50)
    
    # Process each image
    success_count = 0
    for i, img_file in enumerate(image_files, 1):
        # Create output filename (convert to .jpg)
        output_name = img_file.stem + '.jpg'
        output_path = Path(PROCESSED_FOLDER) / output_name
        
        print(f"[{i}/{len(image_files)}] Processing: {img_file.name}...", end=' ')
        
        if resize_image(img_file, output_path, TARGET_SIZE):
            print("✅")
            success_count += 1
        
    print("-" * 50)
    print(f"✨ Done! Successfully processed {success_count}/{len(image_files)} images")
    print(f"📂 Check your images in: {PROCESSED_FOLDER}/")
    
    # Show stats
    if success_count > 0:
        total_size = sum(f.stat().st_size for f in Path(PROCESSED_FOLDER).glob('*.jpg'))
        avg_size = total_size / success_count / 1024  # KB
        print(f"📊 Average file size: {avg_size:.1f} KB")

if __name__ == "__main__":
    print("=" * 50)
    print("🎨 ZEN WALLPAPER DATASET - IMAGE RESIZER")
    print("=" * 50)
    main()