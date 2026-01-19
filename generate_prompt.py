import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import argparse
import os
import csv
from pathlib import Path

# reference: https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html
# Model path
path = "OpenGVLab/InternVL3_5-8B"

# Prompt templates
# ANALYSIS_PROMPT = '''Describe the key feature of this industrial component in a simple sentence, please make the description concise, focus on the shape and holes and the protrusion of the component.'''

ANALYSIS_PROMPT = '''
**Role:** You are an AI assistant generating training data for a CLIP-based industrial inspection model.

**Task:** Based on the geometric attributes of the class (e.g., L-shaped, circular holes, slots), generate **10 distinct, standalone descriptive sentences**.

**Strict Formatting Rules:**

1. **Output Format:** Provide a simple **flat list** (bullet points). No nested lists, no bold headers (like "Topology:").
2. **Sentence Structure:** Each bullet point must be a **complete sentence** following the pattern: *"The component [verb] [adjective] [noun]."* or *"This part features [geometric detail]."*
3. **Content:**
    - Sentence 1-2: Describe the **Global Shape** (e.g., "The object is an L-shaped metal bracket.").
    - Sentence 3-6: Describe **Holes & Slots** individually (e.g., "It contains one large circular hole in the center.", "There is a rectangular slot near the edge.").
    - Sentence 7-8: Describe **Protrusions** (e.g., "The part has raised edges along the sides.").
    - Sentence 9-10: Describe **Relative Positions** (e.g., "Several smaller holes are arranged around the central opening.").
4. **Constraint:** Do NOT mention color, texture, or surface quality (smooth/shiny) to avoid domain shift issues. Focus ONLY on geometry.

**Example Output(please show in numbers sequence):**

1. The industrial part has a rigid L-shaped structure.
2. This component features a large circular hole located centrally.
3. There is a small rectangular slot positioned near the top edge.
4. ...
'''

# GPU device selection
def get_device(gpu_id=None):
    if gpu_id is not None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return f'cuda:{gpu_id}'
        else:
            print(f"GPU {gpu_id} not available, using CPU")
            return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Official recommended dynamic preprocessing functions (Start) ---
def build_transform(input_size=448):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image to fit tile size
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    # Create tiles
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    # Use dynamic preprocessing to split images
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values) # This will become (num_patches, 3, 448, 448)
    return pixel_values
# --- Official recommended dynamic preprocessing functions (End) ---

# Process single image
def process_single_image(model, tokenizer, device, image_path, question, generation_config):
    """Process a single image and return the result"""
    try:
        print(f"Processing image: {os.path.basename(image_path)}")
        
        # Load and process image
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device)
        
        # Generate description
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        print(f"\nResult:\n{response}\n")
        return response
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return None

# Batch process images
def process_batch_images(model, tokenizer, device, input_dir, output_csv, question, generation_config):
    """Process all images in a directory and save results to CSV"""
    
    # Create log file
    log_file = output_csv.replace('.csv', '_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Log started at {device}\n")
        f.write("=" * 50 + "\n")
    
    # Get all image files from input directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Prepare CSV output
    results = []
    
    # Process each image
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load and process image
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device)
            
            # Generate description
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            
            # Extract class name and filename
            relative_path = os.path.relpath(image_path, input_dir)
            class_name = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'Unknown'
            filename = os.path.basename(image_path)
            
            # Store result
            results.append({
                'class_name': class_name,
                'image_file_name': filename,
                'response': response,
                'full_path': image_path
            })
            
            # Log to file immediately
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nImage {i+1}/{len(image_files)}: {filename}\n")
                f.write(f"Class: {class_name}\n")
                f.write(f"Response: {response}\n")
                f.write("-" * 40 + "\n")
            
            print(f"Result: {response[:100]}..." if len(response) > 100 else f"Result: {response}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            relative_path = os.path.relpath(image_path, input_dir)
            class_name = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'Unknown'
            filename = os.path.basename(image_path)
            error_msg = f"Error: {str(e)}"
            results.append({
                'class_name': class_name,
                'image_file_name': filename,
                'response': error_msg,
                'full_path': image_path
            })
            
            # Log error to file immediately
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nImage {i+1}/{len(image_files)}: {filename}\n")
                f.write(f"Class: {class_name}\n")
                f.write(f"ERROR: {str(e)}\n")
                f.write("-" * 40 + "\n")
    
    # Save results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['class_name', 'image_file_name', 'response', 'full_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to {output_csv}")
    print(f"Log saved to {log_file}")
    print(f"Processed {len(results)} images successfully")
    
    # Final log entry
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=" * 50 + "\n")
        f.write(f"Processing completed. Total images: {len(results)}\n")
        f.write(f"CSV saved to: {output_csv}\n")

# Main execution function
def main(mode='batch', gpu_id=None, input_dir=None, image_path=None, output_csv=None, use_analysis_prompt=False):
    device = get_device(gpu_id)
    
    # Load model
    print(f"Loading model on {device}...")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # Select prompt
    question = ANALYSIS_PROMPT
    
    if mode == 'single':
        if not image_path:
            print("Error: --image-path is required for single mode")
            return
        process_single_image(model, tokenizer, device, image_path, question, generation_config)
    elif mode == 'batch':
        if not input_dir:
            print("Error: --input-dir is required for batch mode")
            return
        if not output_csv:
            output_csv = 'image_descriptions.csv'
        process_batch_images(model, tokenizer, device, input_dir, output_csv, question, generation_config)
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'single' or 'batch'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate prompt using InternVL model for images')
    parser.add_argument('--mode', type=str, default='batch', choices=['single', 'batch'],
                        help='Processing mode: single image or batch directory (default: batch)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (e.g., 0, 1, 2). Default is 0')
    parser.add_argument('--input-dir', type=str, default='/mnt/backups/andycw/dataset/M58/Real_all_nobg_augmented', 
                        help='Directory containing images to process (for batch mode)')
    parser.add_argument('--image-path', type=str, help='Path to single image file (for single mode)', default='/mnt/backups/andycw/dataset/M58/CAD_ratioFilter/M58-Encoder側走行機身/12_0.3121959928585598M58-Encoder側走行機身_0Axis_8degree.jpg')
    parser.add_argument('--output-csv', type=str, default='targetImage_descriptions_new.csv', 
                        help='Output CSV file path (for batch mode)')
    parser.add_argument('--use-analysis-prompt', action='store_true',
                        help='Use the detailed analysis prompt for industrial parts')
    args = parser.parse_args()
    
    main(
        mode=args.mode,
        gpu_id=args.gpu,
        input_dir=args.input_dir,
        image_path=args.image_path,
        output_csv=args.output_csv,
        use_analysis_prompt=args.use_analysis_prompt
    )