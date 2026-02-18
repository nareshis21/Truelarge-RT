import os
import sys
from PIL import Image, ImageDraw

def crop_center_circle(img):
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + img.size, fill=255)
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.paste(img, (0, 0), mask=mask)
    return result

def optimize_image(path, target_size_mb=1.0):
    target_bytes = target_size_mb * 1024 * 1024
    img = Image.open(path)
    
    # If it's the app_icon, crop it
    if "app_icon" in path.lower():
        img = crop_center_circle(img)
        print(f"Cropped {path} to circle.")

    quality = 95
    img.save(path, optimize=True, quality=quality)
    
    while os.path.getsize(path) > target_bytes and quality > 10:
        quality -= 5
        img.save(path, optimize=True, quality=quality)
        
    print(f"Optimized {path} to {os.path.getsize(path)/1024/1024:.2f}MB (Quality: {quality})")

def main():
    docs_dir = os.path.join(os.getcwd(), "docs")
    if not os.path.exists(docs_dir):
        print(f"Error: {docs_dir} not found.")
        return

    for filename in os.listdir(docs_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(docs_dir, filename)
            optimize_image(path)

if __name__ == "__main__":
    main()
