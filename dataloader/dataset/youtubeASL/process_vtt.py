import os
import re

def process_vtt_files(video_dir, input_folder, output_file):
    vtt_files = [file_name for file_name in os.listdir(input_folder) if file_name.endswith(".vtt")]
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, vtt_name in enumerate(vtt_files):
            video_name_prefix = vtt_name.split(".")[0]
            video_name = video_name_prefix + ".mp4"
            video_filepath = os.path.join(video_dir, video_name)
            if not os.path.exists(video_filepath):
                print(f"视频文件 {video_filepath} 不存在")
                continue
            print(f"处理第 {i+1} 个文件 {vtt_name}")
            vtt_path = os.path.join(input_folder, vtt_name)

            with open(vtt_path, "r", encoding="utf-8") as vtt_f:
                lines = vtt_f.readlines()
            
            timestamp = None
            text_buffer = []
            
            for line in lines:
                line = line.strip()
                
                # 识别时间戳
                match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", line)
                if match:
                    start_time, end_time = match.groups()  # 获取完整的时间范围
                    # 如果之前有存储的文本，写入文件
                    if timestamp and text_buffer:
                        out_f.write(f"{video_name}||{vtt_name}||{timestamp}||{' '.join(text_buffer)}\n")
                    
                    timestamp = f"{start_time} --> {end_time}"  # 存完整的时间范围
                    text_buffer = []
                elif line and not line.startswith(("WEBVTT", "Kind:", "Language:")):
                    text_buffer.append(line)
            
            # 处理最后一段文本
            if timestamp and text_buffer:
                out_f.write(f"{video_name}||{vtt_name}||{timestamp}||{' '.join(text_buffer)}\n")
            
            # break


video_dir = "/scratch/rhong5/dataset/youtube_ASL"  # 替换为视频文件所在的文件夹路径
input_folder = "/scratch/rhong5/dataset/youtube_ASL"  # 替换为 .vtt 文件所在的文件夹路径
output_file = "/scratch/rhong5/dataset/youtubeASL_annotation.txt"  # 输出文件路径
if os.path.exists(output_file):
    os.remove(output_file)
process_vtt_files(video_dir, input_folder, output_file)
print("处理完成，结果已保存到", output_file)
