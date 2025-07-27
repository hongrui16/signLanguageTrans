import os
import re
import shutil
import json
import argparse


def parse_timestamp(ts: str) -> float:
    """将 00:01:09.249 转换为秒"""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def process_txt_file(txt_path, output_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]


    kept_lines = []

    for line in lines:
        
        if not line or ' --> ' not in line:
            continue

        try:
            video_name, vtt_file, time_range, text = line.split("||")
            start_str, end_str = time_range.strip().split(" --> ")
            start_time = parse_timestamp(start_str)
            end_time = parse_timestamp(end_str)
            duration = end_time - start_time
        except ValueError:
            print(f"跳过格式错误的行: {line}")
            continue

        # 1. 时长 < 1 秒 → 丢弃
        if duration < 1.0:
            continue

        # 提取 < 前文本
        text_before_tag = re.split(r"<", text)[0].strip()
        tokens = text_before_tag.split()

        # 2. 只有一个词（无论是否含 <） → 丢弃
        if len(tokens) == 1:
            continue

        # 3. 含多个词但有 <，保留 < 前的所有词（不含最后一个）
        if "<" in text and len(tokens) > 1:
            new_tokens = tokens[:-1]  # 去掉最后一个词
            if not new_tokens:
                continue  # 如果结果为空，也丢弃
            new_text = " ".join(new_tokens)
            updated_line = f"{video_name}||{vtt_file}||{time_range}||{new_text}"
            kept_lines.append(updated_line)
        else:
            # 保留原始行
            kept_lines.append(line)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
            

def read_txt_process_json(txt_path, json_dir, output_dir, start_index=0, end_index=None):
    
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    if end_index is None:
        end_index = len(lines)
    elif end_index > len(lines):
        end_index = len(lines)
        
    temp_lines = lines[start_index:end_index]
    for i, line in enumerate(temp_lines):
        print(f"Processing line {i+start_index}/{end_index}")

        clip_id = f'{i+start_index:08d}'

        if not line or ' --> ' not in line:
            continue

        try:
            video_name, vtt_file, time_range, text = line.split("||")
            start_str, end_str = time_range.strip().split(" --> ")
            start_time = parse_timestamp(start_str)
            end_time = parse_timestamp(end_str)
            duration = end_time - start_time
        except ValueError:
            print(f"跳过格式错误的行: {line}")
            continue
            
        video_name_prefix = video_name.split('.')[0]
        
        json_filepath = os.path.join(json_dir, f'{video_name_prefix}_SID{clip_id}_anno.json')
        if not os.path.exists(json_filepath):
            print(f"Warning: {json_filepath} does not exist.")
            continue
        new_json_filepath = os.path.join(output_dir, f'{video_name_prefix}_SID{clip_id}_anno.json')
        
        # 1. 时长 < 1 秒 → 丢弃
        if duration < 1.0:
            shutil.move(json_filepath, new_json_filepath)
            continue

        # 提取 < 前文本
        text_before_tag = re.split(r"<", text)[0].strip()
        tokens = text_before_tag.split()

        # 2. 只有一个词（无论是否含 <） → 丢弃
        if len(tokens) == 1:
            shutil.move(json_filepath, new_json_filepath)
            continue

        # 3. 含多个词但有 <，保留 < 前的所有词（不含最后一个）
        if "<" in text and len(tokens) > 1:            
            new_tokens = tokens[:-1]  # 去掉最后一个词
            if not new_tokens:
                shutil.move(json_filepath, new_json_filepath)
                continue  # 如果结果为空，也丢弃
            new_text = " ".join(new_tokens)
            with open(json_filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[跳过无效 JSON] {json_filepath}")
                    continue
            data["text"] = new_text
            shutil.move(json_filepath, new_json_filepath)
            
            # dump the updated JSON
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            


if __name__ == "__main__":
    # input_txt = "/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/old_anno.txt"     # 修改为你的输入文件名
    # output_txt = "/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/filtered_output.txt"
    # process_txt_file(input_txt, output_txt)
    # print(f"✅ 已保存清洗后的字幕到: {output_txt}")
    start_index =  342040

    read_txt_process_json(
        txt_path="/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_annotation.txt",
        json_dir="/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/annos",
        output_dir="/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/wrong_annos",
        start_index=start_index
    )