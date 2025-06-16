
import pandas as pd
import re
import os, sys

def parse_csv():
    split = 'val'  # 可以根据需要修改为 'train' 或 'val'
    csv_path = f'/projects/kosecka/hongrui/dataset/how2sign/re-aligned_how2sign_realigned_{split}.csv'  # 替换成你的文件路径
    output_path = f'/projects/kosecka/hongrui/dataset/how2sign/re-aligned_how2sign_realigned_{split}.txt'  # 替换成你的输出文件路径
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        os.remove(output_path)  # 删除已存在的输出文件
    print(f"Processing CSV file: {csv_path}")
    print(f"Output will be saved to: {output_path}")
    # 读取CSV文件
    df = pd.read_csv(csv_path, sep='\t')  # 注意分隔符是 tab，或根据你实际保存的分隔符调整
    
    # 提取 SENTENCE_ID 中的数字索引
    def extract_index(s):
        match = re.search(r'_(\d+)$', s)
        return int(match.group(1)) if match else -1

    df["INDEX"] = df["SENTENCE_ID"].apply(extract_index)

    # 在每个 VIDEO_NAME 内部排序，并添加本地的两位数编号
    df_sorted = df.sort_values(by=["VIDEO_NAME", "INDEX"]).reset_index(drop=True)

    # 分组并生成本地编号（两位数左补零）
    df_sorted["SENTENCE_ID"] = (
        df_sorted.groupby("VIDEO_NAME").cumcount()
    ).apply(lambda x: f"{x:03d}")


    # 选择并保存需要的列
    lines = df_sorted[["VIDEO_NAME", "SENTENCE_ID", "START_REALIGNED", "END_REALIGNED", "SENTENCE"]].values.tolist()
    
    # 写入 txt 文件，无标题
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("\t".join(map(str, line)) + "\n")



def print_csv_content():
    csv_path = '/projects/kosecka/hongrui/dataset/how2sign/re-aligned_how2sign_realigned_test.csv'  # 替换成你的文件路径

    video_names = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split()  # 使用空格拆分
            if len(fields) >= 7:
                video_id = fields[0]
                video_file = fields[1]
                clip_id = fields[2]
                clip_file = fields[3]
                # start_time = float(fields[4])
                # end_time = float(fields[5])
                caption = ' '.join(fields[6:])  # 余下是 caption
                # print(f"Video ID: {video_id}")
                # print(f"Clip ID: {clip_id}")
                # print(f"Start-End: {start_time}-{end_time}")
                # print(f"Caption: {caption}")
                # print("---")

                video_names.add(video_file)  # 添加视频文件名到集合中

        



    print('from csv', len(video_names))  # 打印集合的大小


    video_dir = '/projects/kosecka/hongrui/dataset/how2sign/video_level/train/rgb_front/raw_videos/'

    video_names = list(video_names)
    print('video_names', len(video_names))
    
if __name__ == "__main__":
    parse_csv()  # 如果需要解析 CSV 文件，可以取消注释
    # print_csv_content()  # 打印 CSV 内容