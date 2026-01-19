import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import csv
from collections import defaultdict
import os
from tqdm import tqdm

# Model path
path = "OpenGVLab/InternVL3_5-8B"

# Prompt template
ANALYSIS_PROMPT = '''
Role: You are an expert mechanical engineer. Your task is to analyze 100 or less raw visual descriptions of industrial parts and distill them into a concise, non-redundant set of geometric attributes.

Task: Analyze the provided 100 or less descriptions. Many descriptions will repeat the same information; your goal is to merge duplicates and identify distinct geometric features into 4 categories.

Four categories:
1.Global Shape: Overall geometry and main body structure (e.g., L-shaped, rigid bracket).
2.Individual Holes & Slots: Details about circular holes, rectangular cutouts, or elongated slots.
3.Protrusions: Features that stand out from the surface, such as raised edges or protruding tabs.
4.Relative Positions: The spatial arrangement of these features relative to each other.

Constraint:
No Repetition: Each bullet point must describe a unique feature. Do not list the same feature multiple times even if it appears in many raw descriptions.
Limit Quantity: Provide no more than 5 to 8 high-quality bullet points per category.
Synthesis: If different descriptions mention the same hole at slightly different positions, synthesize them into one representative point.
Formatting: Use the following headers exactly: [GLOBAL_SHAPE], [HOLES_SLOTS], [PROTRUSIONS], and [RELATIVE_POSITIONS].

Formatting Rules: 
1. Use the exactly following four headers: [GLOBAL_SHAPE], [HOLES_SLOTS], [PROTRUSIONS], and [RELATIVE_POSITIONS]. 
2. List items using bullet points (-) only. 
3. Do not include sub-headings or introductory text.

Example Output: 
[GLOBAL_SHAPE]
L-shaped metal bracket configuration.
Main body features a rigid structure.

[HOLES_SLOTS]
One large circular hole located centrally.
Multiple small rectangular slots near the edges.
... (continue for other categories)

Example of Good Synthesis:
Raw: "Small hole on left", "Tiny hole on left side", "Circular opening on left".
Your Output: "- One small circular hole is located on the left edge."
'''


def get_device(gpu_id=None):
    """Select GPU device"""
    if gpu_id is not None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return f'cuda:{gpu_id}'
        else:
            print(f"GPU {gpu_id} not available, using CPU")
            return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_response_sentences(response):
    """Parse response into sentences and split into 4 categories"""
    # Split by newline and filter empty lines
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    # Extract sentences (assuming they start with numbers like "1.", "2.", etc.)
    sentences = []
    for line in lines:
        # Check if line starts with a number followed by a period or dot
        if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
            sentences.append(line)
    
    # If we don't have exactly 10 sentences, just take the first 10 lines
    if len(sentences) < 10:
        sentences = lines[:10] if len(lines) >= 10 else lines
    
    # Categorize sentences
    categories = {
        'Global Shape': sentences[0:2] if len(sentences) >= 2 else sentences[0:1],
        'Holes & Slots': sentences[2:6] if len(sentences) >= 6 else sentences[2:len(sentences)],
        'Protrusions': sentences[6:8] if len(sentences) >= 8 else [],
        'Relative Positions': sentences[8:10] if len(sentences) >= 10 else []
    }
    
    return categories


def format_batch_descriptions(batch_data):
    """Format a batch of image descriptions into categorized text"""
    # Collect all sentences by category
    all_categories = {
        'Global Shape': [],
        'Holes & Slots': [],
        'Protrusions': [],
        'Relative Positions': []
    }
    
    # Parse all images and collect sentences by category
    for item in batch_data:
        categories = parse_response_sentences(item['response'])
        for cat_name in all_categories.keys():
            all_categories[cat_name].extend(categories[cat_name])
    
    # Format the output
    formatted_text = f"Below are descriptions from {len(batch_data)} images of the same component:\n\n"
    
    formatted_text += "**Global Shape:**\n"
    for sentence in all_categories['Global Shape']:
        # Remove the number prefix if present
        clean_sentence = sentence.lstrip('0123456789. ')
        formatted_text += f"{clean_sentence}\n"
    
    formatted_text += "\n**Holes & Slots:**\n"
    for sentence in all_categories['Holes & Slots']:
        clean_sentence = sentence.lstrip('0123456789. ')
        formatted_text += f"{clean_sentence}\n"
    
    formatted_text += "\n**Protrusions:**\n"
    for sentence in all_categories['Protrusions']:
        clean_sentence = sentence.lstrip('0123456789. ')
        formatted_text += f"{clean_sentence}\n"
    
    formatted_text += "\n**Relative Positions:**\n"
    for sentence in all_categories['Relative Positions']:
        clean_sentence = sentence.lstrip('0123456789. ')
        formatted_text += f"{clean_sentence}\n"
    
    return formatted_text


def parse_llm_response(response):
    """Parse LLM response to extract the four categories"""
    categories = {
        'global_shape': [],
        'holes_slots': [],
        'protrusions': [],
        'relative_positions': []
    }
    
    # Split response into lines
    lines = response.split('\n')
    current_category = None
    
    for line in lines:
        line = line.strip()
        
        # Check for category headers
        if '[GLOBAL_SHAPE]' in line:
            current_category = 'global_shape'
        elif '[HOLES_SLOTS]' in line:
            current_category = 'holes_slots'
        elif '[PROTRUSIONS]' in line:
            current_category = 'protrusions'
        elif '[RELATIVE_POSITIONS]' in line:
            current_category = 'relative_positions'
        elif line and current_category:
            # Remove leading dash and spaces if present
            clean_line = line.lstrip('- ').strip()
            if clean_line:
                # Append to current category as a separate item
                categories[current_category].append(clean_line)
    
    # Join items with newline for CSV display
    return {
        'global_shape': '\n'.join(categories['global_shape']),
        'holes_slots': '\n'.join(categories['holes_slots']),
        'protrusions': '\n'.join(categories['protrusions']),
        'relative_positions': '\n'.join(categories['relative_positions'])
    }


def load_csv_data(csv_path):
    """Load CSV data and group by class_name"""
    data_by_class = defaultdict(list)
    
    # Use 'utf-8-sig' to handle BOM (Byte Order Mark) if present
    with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip empty rows
            if not row or 'class_name' not in row:
                continue
            
            class_name = row['class_name']
            data_by_class[class_name].append({
                'class_name': row['class_name'],
                'image_file_name': row['image_file_name'],
                'response': row['response'],
                'full_path': row['full_path']
            })
    
    return data_by_class


def process_class_in_batches(model, tokenizer, device, class_name, class_data, batch_size=10, output_file=None, csv_output_file=None):
    """Process one class in batches of images"""
    
    # Prepare output file
    if output_file is None:
        output_file = f"reduction_output_{class_name.replace('/', '_')}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Reduction Phase Results for Class: {class_name}\n")
        f.write(f"Total Images: {len(class_data)}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write("="*70 + "\n\n")
    
    # Count processed batches
    batch_count = 0
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # Process in batches
    num_batches = (len(class_data) + batch_size - 1) // batch_size
    
    # Use tqdm for progress bar
    for batch_idx in tqdm(range(num_batches), desc=f"  Batches", unit="batch", leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(class_data))
        batch_data = class_data[start_idx:end_idx]
        
        # Format the batch descriptions
        formatted_descriptions = format_batch_descriptions(batch_data)
        
        # Create the full question with analysis prompt
        question = ANALYSIS_PROMPT + "\n\n" + formatted_descriptions
        
        try:
            # Pure-text conversation (no image input, pixel_values=None)
            response, history = model.chat(
                tokenizer, 
                None,  # No image input for pure text
                question, 
                generation_config, 
                history=None, 
                return_history=True
            )
            
            # Parse LLM response
            parsed_categories = parse_llm_response(response)
            csv_row = {
                'class_name': class_name,
                'batch_idx': batch_idx + 1,
                'global_shape': parsed_categories['global_shape'],
                'holes_slots': parsed_categories['holes_slots'],
                'protrusions': parsed_categories['protrusions'],
                'relative_positions': parsed_categories['relative_positions']
            }
            
            # Write to CSV immediately
            if csv_output_file:
                with open(csv_output_file, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['class_name', 'batch_idx', 'global_shape', 'holes_slots', 'protrusions', 'relative_positions'])
                    writer.writerow(csv_row)
            
            batch_count += 1
            
            # Save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"BATCH {batch_idx + 1}/{num_batches} (Images {start_idx + 1}-{end_idx})\n")
                f.write(f"{'='*70}\n\n")
                f.write("INPUT DESCRIPTIONS:\n")
                f.write("-"*70 + "\n")
                f.write(formatted_descriptions)
                f.write("\n" + "-"*70 + "\n")
                f.write("MODEL RESPONSE:\n")
                f.write("-"*70 + "\n")
                f.write(response)
                f.write("\n" + "-"*70 + "\n\n")
        
        except Exception as e:
            # Write error row to CSV immediately
            if csv_output_file:
                error_row = {
                    'class_name': class_name,
                    'batch_idx': batch_idx + 1,
                    'global_shape': f'ERROR: {str(e)}',
                    'holes_slots': '',
                    'protrusions': '',
                    'relative_positions': ''
                }
                with open(csv_output_file, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['class_name', 'batch_idx', 'global_shape', 'holes_slots', 'protrusions', 'relative_positions'])
                    writer.writerow(error_row)
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"BATCH {batch_idx + 1}/{num_batches} - ERROR\n")
                f.write(f"{'='*70}\n")
                f.write(f"ERROR: {str(e)}\n\n")
    
    return output_file, batch_count


def main(csv_path, gpu_id=0, batch_size=10, output_dir='reduction_results'):
    """Main function - Process all classes"""
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
    print("Model loaded successfully!\n")
    
    # Load CSV data
    print(f"Loading data from: {csv_path}")
    data_by_class = load_csv_data(csv_path)
    print(f"Found {len(data_by_class)} classes\n")
    
    # List all classes
    sorted_classes = sorted(data_by_class.keys())
    print(f"Total classes to process: {len(sorted_classes)}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare CSV output file
    csv_output_file = os.path.join(output_dir, 'reduction_results.csv')
    
    # Create CSV file with headers
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['class_name', 'batch_idx', 'global_shape', 'holes_slots', 'protrusions', 'relative_positions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Process all classes with progress bar
    total_classes = len(sorted_classes)
    print("Starting batch processing...\n")
    for class_idx, class_name in enumerate(sorted_classes, 1):
        print(f"\n[{class_idx}/{total_classes}] Class: {class_name} ({len(data_by_class[class_name])} images)")
        
        class_data = data_by_class[class_name]
        output_file = os.path.join(output_dir, f"reduction_{class_name.replace('/', '_')}.txt")
        
        try:
            txt_file, batch_count = process_class_in_batches(
                model, tokenizer, device, class_name, class_data, batch_size, output_file, csv_output_file
            )
            
            print(f"✓ Completed ({batch_count} batches written to CSV)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            # Log error to CSV
            with open(csv_output_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['class_name', 'batch_idx', 'global_shape', 'holes_slots', 'protrusions', 'relative_positions']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'class_name': class_name,
                    'batch_idx': 0,
                    'global_shape': f'ERROR: {str(e)}',
                    'holes_slots': '',
                    'protrusions': '',
                    'relative_positions': ''
                })
    
    print(f"\n{'='*70}")
    print(f"✓ ALL CLASSES COMPLETED!")
    print(f"  Total classes processed: {total_classes}")
    print(f"  CSV results: {csv_output_file}")
    print(f"  Text files: {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduction phase: Process all classes with batch processing')
    parser.add_argument('--csv', type=str, 
                        default='/mnt/backups/andycw/internvl/image_description/M58_79classes_sourceImage_descriptions.csv',
                        help='Path to CSV file containing image descriptions')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--batch-size', type=int, default=15,
                        help='Number of images per batch (default: 10)')
    parser.add_argument('--output-dir', type=str, default='reduction_results',
                        help='Directory to save output files (default: reduction_results)')
    
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
