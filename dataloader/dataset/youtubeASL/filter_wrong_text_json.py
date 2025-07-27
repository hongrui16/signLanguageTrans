import os
import json
import shutil
import re

def process_json_folder(json_dir, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

    for i, filename in enumerate(os.listdir(json_dir)):
        print(f"Processing {i}/{len(os.listdir(json_dir))}: {filename}")
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[跳过无效 JSON] {filename}")
                continue

        text = data.get("text", "").strip()
        start_time = data.get("start_time", 0)
        end_time = data.get("end_time", 0)
        duration = end_time - start_time

        # 1. 如果持续时间 < 1 秒
        if duration < 1.0:
            shutil.move(filepath, os.path.join(temp_dir, filename))
            continue

        # 提取 < 前面的部分
        text_before_tag = re.split(r"<", text)[0].strip()
        tokens = text_before_tag.split()

        # 2. 只有一个词并且后面紧跟 <，如：familiar<...
        if len(tokens) == 1:
            shutil.move(filepath, os.path.join(temp_dir, filename))
            print(f"move {filepath} - only one word ")
            continue

        # # 3. 只有一个词，无 <
        # if len(tokens) == 1 and "<" not in text:
        #     shutil.move(filepath, os.path.join(temp_dir, filename))
        #     print(f"move {filepath} - only one word")
        #     continue

        # 4. 有多个词，且含 <，保留 < 前的所有词，**不含最后一个**
        if "<" in text and len(tokens) > 1:
            new_tokens = tokens[:-1]  # 去掉最后一个词
            if new_tokens:
                new_text = " ".join(new_tokens)
                data["text"] = new_text
                os.remove(filepath)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"Updated {filepath} - kept text before < and removed last word")
            else:
                # 如果删掉最后一个后变成空的，也移动到 temp
                shutil.move(filepath, os.path.join(temp_dir, filename))

                


if __name__ == "__main__":
    json_folder = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/annos'
    temp_folder = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/wrong_annos'
    process_json_folder(json_folder, temp_folder)
