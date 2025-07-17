import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os



json_txt_path = '/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/train/train_annos_filepath.txt'
new_json_txt_path = '/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/train/train_annos_filepath_filtered.txt'



# 区间定义
bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100),
        (100, 120), (120, 150), (150, 180), (180, float('inf'))]
bin_labels = ["0~20", "20~40", "40~60", "60~80", "80~100",
              "100~120", "120~150", "150~180", "180~210", "210~240", "240~270", ">270"]

# 初始化计数字典
distribution = defaultdict(int)

# 读取txt中的json路径
with open(json_txt_path, 'r') as f:
    json_paths = [line.strip() for line in f if line.strip()]

# 保存有效路径
filtered_json_paths = []

# 遍历每个json文件
for i, json_path in enumerate(json_paths):
    print(f"Processing {i + 1}/{len(json_paths)}: {json_path}")
    if not os.path.exists(json_path):
        print(f"[Warning] File not found: {json_path}")
        continue

    try:
        with open(json_path, 'r') as jf:
            clip_info_dict = json.load(jf)
    except json.JSONDecodeError:
        print(f"[Warning] JSON decode error in file: {json_path}")
        continue

    keyframe_count = len(clip_info_dict.get('keyframes', {}))

    if keyframe_count < 5:
        continue  # 跳过 keyframe 少于 5 的 clip

    filtered_json_paths.append(json_path)

    # 统计分布
    for (low, high), label in zip(bins, bin_labels):
        if low <= keyframe_count < high:
            distribution[label] += 1
            break

# 保存新的 txt 文件
with open(new_json_txt_path, 'w') as f:
    for path in filtered_json_paths:
        f.write(path + '\n')

print(f"\nSaved filtered list to {new_json_txt_path} (total: {len(filtered_json_paths)})")

# 可视化
counts = [distribution[label] for label in bin_labels]

plt.figure(figsize=(10, 6))
plt.bar(bin_labels, counts)
plt.xlabel("Keyframe Count Interval")
plt.ylabel("Number of Clips")
plt.title("How2sign Distribution of Keyframe Counts in Clips (≥5 frames)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("keyframe_distribution.png")
plt.close()
