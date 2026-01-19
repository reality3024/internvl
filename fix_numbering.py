import csv
import re

def fix_bullet_to_numbering(text):
    """
    將 "- " 開頭的列表項目轉換為數字編號格式 "1. 2. 3. ..."
    """
    lines = text.split('\n')
    fixed_lines = []
    counter = 1
    
    for line in lines:
        # 檢查是否以 "- " 開頭
        if line.strip().startswith('- '):
            # 移除 "- " 並加上數字編號
            content = line.strip()[2:]  # 移除 "- "
            fixed_lines.append(f"{counter}. {content}")
            counter += 1
        else:
            fixed_lines.append(line)
            # 如果遇到非 "- " 開頭的行，重置計數器
            if line.strip() and not line.strip().startswith('- '):
                counter = 1
    
    return '\n'.join(fixed_lines)

def main():
    input_file = 'sourceImage_descriptions_new.csv'
    output_file = 'sourceImage_descriptions_fixed.csv'
    
    fixed_count = 0
    total_count = 0
    
    # 讀取並處理 CSV
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        rows = []
        for row in reader:
            total_count += 1
            original_response = row['response']
            
            # 檢查是否包含 "- " 格式
            if '\n- ' in original_response or original_response.strip().startswith('- '):
                fixed_response = fix_bullet_to_numbering(original_response)
                row['response'] = fixed_response
                fixed_count += 1
            
            rows.append(row)
    
    # 寫入新的 CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"處理完成！")
    print(f"總共處理了 {total_count} 筆資料")
    print(f"修正了 {fixed_count} 筆有問題的編號格式")
    print(f"結果已儲存至: {output_file}")

if __name__ == '__main__':
    main()
