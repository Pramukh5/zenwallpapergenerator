import os
import glob
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Configuration ---

# 1. Your required keywords to add to every caption.
STYLE_PREFIX = "Zen, minimal, abstract composition. "
STYLE_SUFFIX = " Creates a wabi sabi feeling."

# 2. The AI model to use for generating the base description.
MODEL_ID = "Salesforce/blip-image-captioning-base"

# 3. How long should the AI's *description* part be?
MAX_DESCRIPTION_LENGTH = 15

# 4. The name of the single output file
OUTPUT_FILENAME = "all_captions.txt"

# --- End of Configuration ---

def setup_model():
    """Loads the AI model and processor from Hugging Face."""
    print(f"Loading model: {MODEL_ID}...")
    print("This may take a few minutes on the first run as the model is downloaded.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    
    print("Model loaded successfully.")
    return processor, model, device

def generate_caption(image_path, processor, model, device):
    """Generates a caption for a single image."""
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, 
            max_new_tokens=MAX_DESCRIPTION_LENGTH, 
            do_sample=False
        )
        description = processor.decode(out[0], skip_special_tokens=True)

        if description.startswith("a photo of") or description.startswith("a picture of"):
             description = description.split(" ", 2)[-1]

        final_caption = f"{STYLE_PREFIX}{description}{STYLE_SUFFIX}"
        return final_caption

    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None

def process_images_in_folder():
    """Finds all JPEGs, captions them, and saves to a *single* text file."""
    
    processor, model, device = setup_model()
    
    image_files = glob.glob("*.jpg")
    
    if not image_files:
        print("Error: No .jpg files found in this directory.")
        print("Please place this script in the same folder as your images.")
        return

    print(f"\nFound {len(image_files)} images to caption...")

    # --- KEY CHANGE: Create a list to hold all caption lines ---
    all_captions_list = []

    for image_path in image_files:
        print(f"Processing {image_path}...")
        
        caption = generate_caption(image_path, processor, model, device)
        
        if caption:
            # Get just the filename (e.g., "zen_001.jpg")
            file_name = os.path.basename(image_path)
            
            # Format the line as you requested
            caption_line = f"{file_name} → {caption}"
            
            # --- KEY CHANGE: Add the line to our list ---
            all_captions_list.append(caption_line)
            print(f"  → Generated caption for {file_name}")

    # --- KEY CHANGE: Write the entire list to one file ---
    print(f"\nWriting all captions to {OUTPUT_FILENAME}...")
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            # Join all lines in the list with a newline character
            f.write("\n".join(all_captions_list))
        print("All images captioned successfully.")
    except IOError as e:
        print(f"  Error writing file {OUTPUT_FILENAME}: {e}")

# --- Run the script ---
if __name__ == "__main__":
    process_images_in_folder()