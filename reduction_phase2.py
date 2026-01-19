import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import csv
from collections import defaultdict
from tqdm import tqdm

# Model path
path = "OpenGVLab/InternVL3_5-8B"

# Prompt template - 参考 reduction_phase2.py 的 ANALYSIS_PROMPT
ANALYSIS_PROMPT = '''
Role: You are a professional mechanical specialist. Your goal is to synthesize multiple structural observations into a definitive set of visual descriptors for a part's class-level prototype.

Task: Below are summarized structural key points extracted from a pool of descriptions for a single part category. Synthesize these points into exactly 10 concise and professional visual descriptions.

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


def get_device(gpu_id=None):
    """選擇 GPU 設備"""
    if gpu_id is not None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return f'cuda:{gpu_id}'
        else:
            print(f"GPU {gpu_id} 不可用，使用 CPU")
            return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_csv_data(csv_path):
    """載入 CSV 資料並按 class_name 分組"""
    data_by_class = defaultdict(list)
    
    # 使用 'utf-8-sig' 處理 BOM (Byte Order Mark)
    with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 跳過空行
            if not row or 'class_name' not in row:
                continue
            
            class_name = row['class_name']
            data_by_class[class_name].append({
                'batch_idx': row['batch_idx'],
                'global_shape': row['global_shape'],
                'holes_slots': row['holes_slots'],
                'protrusions': row['protrusions'],
                'relative_positions': row['relative_positions']
            })
    
    return data_by_class


def format_class_descriptions(class_name, class_data):
    """將一個類別的所有批次格式化為分類文本
    類似 reduction_phase.py 的 format_batch_descriptions() 但處理的是已經分類好的資料
    """
    formatted_text = f"Below are summarized structural key points extracted from multiple batches for the part category: {class_name}\n\n"
    
    formatted_text += "**Global Shape:**\n"
    for batch in class_data:
        if batch['global_shape'].strip():
            # 按換行符分割並添加每一行
            for line in batch['global_shape'].split('\n'):
                if line.strip():
                    formatted_text += f"{line.strip()}\n"
    
    formatted_text += "\n**Holes & Slots:**\n"
    for batch in class_data:
        if batch['holes_slots'].strip():
            for line in batch['holes_slots'].split('\n'):
                if line.strip():
                    formatted_text += f"{line.strip()}\n"
    
    formatted_text += "\n**Protrusions:**\n"
    for batch in class_data:
        if batch['protrusions'].strip():
            for line in batch['protrusions'].split('\n'):
                if line.strip():
                    formatted_text += f"{line.strip()}\n"
    
    formatted_text += "\n**Relative Positions:**\n"
    for batch in class_data:
        if batch['relative_positions'].strip():
            for line in batch['relative_positions'].split('\n'):
                if line.strip():
                    formatted_text += f"{line.strip()}\n"
    
    return formatted_text


def process_all_classes(model, tokenizer, device, data_by_class, output_csv, log_file=None):
    """處理所有類別並生成最終描述"""
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # 建立 CSV 文件並寫入標題
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['class_name', 'num_batches_synthesized', 'final_description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # 建立日誌文件
    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"最終合成日誌\n")
            f.write(f"處理設備：{device}\n")
            f.write("=" * 70 + "\n\n")
    
    # 取得排序後的類別名稱
    sorted_classes = sorted(data_by_class.keys())
    total_classes = len(sorted_classes)
    processed_count = 0
    
    print(f"\n開始對 {total_classes} 個類別進行最終合成...\n")
    
    # 處理每個類別
    for class_idx, class_name in enumerate(tqdm(sorted_classes, desc="處理類別中", unit="類別")):
        try:
            class_data = data_by_class[class_name]
            num_batches = len(class_data)
            
            # 格式化此類別的描述
            formatted_descriptions = format_class_descriptions(class_name, class_data)
            
            # 建立包含分析提示的完整問題
            question = ANALYSIS_PROMPT + "\n\n" + formatted_descriptions
            
            # 純文本對話（無圖像輸入，pixel_values=None）
            response, history = model.chat(
                tokenizer, 
                None,  # 純文本無圖像輸入
                question, 
                generation_config, 
                history=None, 
                return_history=True
            )
            
            # 立即儲存結果到 CSV
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['class_name', 'final_description']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'class_name': class_name,
                    'final_description': response
                })
            
            processed_count += 1
            
            # 清理 CUDA 快取以釋放記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 立即記錄到檔案
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*70}\n")
                    f.write(f"類別 {class_idx + 1}/{total_classes}: {class_name}\n")
                    f.write(f"已合成批次數: {num_batches}\n")
                    f.write(f"{'='*70}\n\n")
                    f.write("輸入:\n")
                    f.write("-"*70 + "\n")
                    f.write(formatted_descriptions)
                    f.write("\n" + "-"*70 + "\n")
                    f.write("回應:\n")
                    f.write("-"*70 + "\n")
                    f.write(response)
                    f.write("\n" + "-"*70 + "\n\n")
            
            print(f"  ✓ {class_name}: 已生成 {len(response)} 個字元")
            
        except Exception as e:
            error_msg = f"錯誤: {str(e)}"
            print(f"  ✗ {class_name}: {error_msg}")
            
            # 立即儲存錯誤結果到 CSV
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['class_name', 'final_description']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'class_name': class_name,
                    'final_description': error_msg
                })
            
            # 立即記錄錯誤到檔案
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*70}\n")
                    f.write(f"類別 {class_idx + 1}/{total_classes}: {class_name}\n")
                    f.write(f"錯誤: {str(e)}\n")
                    f.write(f"{'='*70}\n\n")
            
            # 清理 CUDA 快取以釋放記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"✓ 處理完成！")
    print(f"  總類別數: {total_classes}")
    print(f"  成功處理: {processed_count} 個類別")
    print(f"  CSV 已儲存至: {output_csv}")
    if log_file:
        print(f"  日誌已儲存至: {log_file}")
    print(f"{'='*70}\n")


def main(csv_path, gpu_id=0, output_csv='final_synthesis_results.csv', output_log='final_synthesis_log.txt'):
    """主函數"""
    device = get_device(gpu_id)
    
    # 載入模型
    print(f"正在載入模型到 {device}...")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print("模型載入成功！\n")
    
    # 載入 CSV 資料
    print(f"正在載入資料：{csv_path}")
    data_by_class = load_csv_data(csv_path)
    print(f"找到 {len(data_by_class)} 個類別\n")
    
    # 顯示統計資訊
    print("類別統計：")
    for class_name in sorted(data_by_class.keys())[:5]:  # 顯示前 5 個
        print(f"  - {class_name}: {len(data_by_class[class_name])} 個批次")
    if len(data_by_class) > 5:
        print(f"  ... 以及其他 {len(data_by_class) - 5} 個類別")
    print()
    
    # 處理所有類別
    process_all_classes(model, tokenizer, device, data_by_class, output_csv, output_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='最終合成：合併每個類別的所有批次結果並生成最終描述'
    )
    parser.add_argument('--csv', type=str, 
                        default='/mnt/backups/andycw/internvl/reduction_results/reduction_results.csv',
                        help='包含批次縮減結果的 CSV 檔案路徑')
    parser.add_argument('--gpu', type=int, default=0,
                        help='要使用的 GPU ID（預設：0）')
    parser.add_argument('--output-csv', type=str, default='final_synthesis_results.csv',
                        help='輸出 CSV 檔案路徑（預設：final_synthesis_results.csv）')
    parser.add_argument('--output-log', type=str, default='final_synthesis_log.txt',
                        help='輸出日誌檔案路徑（預設：final_synthesis_log.txt）')
    
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        gpu_id=args.gpu,
        output_csv=args.output_csv,
        output_log=args.output_log
    )
