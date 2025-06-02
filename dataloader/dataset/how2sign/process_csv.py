


csv_path = '/projects/kosecka/hongrui/dataset/how2sign/how2sign_realigned_train.csv'  # 替换成你的文件路径

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