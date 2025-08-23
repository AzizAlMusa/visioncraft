from PIL import Image
import os

# Input and output directories
input_dir = "./images_raw"     # folder with your original renders
output_dir = "./images_cropped"
os.makedirs(output_dir, exist_ok=True)

# Target square size (set to min dimension so you don't distort)
TARGET_SIZE = 600  # adjust as needed (e.g., 512, 800)

def center_square_crop(img, size):
    w, h = img.size
    min_dim = min(w, h)
    # crop a centered square
    left   = (w - min_dim) // 2
    top    = (h - min_dim) // 2
    right  = left + min_dim
    bottom = top + min_dim
    img_cropped = img.crop((left, top, right, bottom))
    # resize to fixed square size
    return img_cropped.resize((size, size), Image.LANCZOS)

for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_dir, fname)
        img = Image.open(path)
        
        cropped = center_square_crop(img, TARGET_SIZE)
        
        save_path = os.path.join(output_dir, fname.replace(".png", "_sq.png"))
        cropped.save(save_path)
        print(f"[Saved] {save_path}")
