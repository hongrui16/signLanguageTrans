import os
import time
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

# 读取 txt 文件中的 YouTube ID
def read_video_ids(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as file:
        video_ids = [line.strip() for line in file if line.strip()]
    return video_ids

# 记录成功的视频 ID（视频 + 字幕都下载成功）
def log_successful_id(video_id, succeed_log_file="successful_ids.txt"):
    with open(succeed_log_file, 'a', encoding='utf-8') as f:
        f.write(f"{video_id}\n")

# 记录失败的视频 ID（视频或字幕缺失）
def log_failed_id(video_id, message, failed_log_file="failed_ids.txt"):
    with open(failed_log_file, 'a', encoding='utf-8') as f:
        f.write(f"{video_id}: {message}\n")

# 下载视频和字幕/转录

subtitleslangs = ['en', 'ase', 'en-CA', 'en-US', 'en-GB', 'en-AU']
def download_video_and_subtitles(video_id, output_dir="downloads", succeed_log_file="successful_ids.txt", failed_log_file="failed_ids.txt"):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # yt-dlp 配置
        ydl_opts = {
            'outtmpl': f'{output_dir}/{video_id}.%(ext)s',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': subtitleslangs,  # 尝试多种语言
            # 'cookiefile': 'www.youtube.com_cookies.txt',  # 如果需要登录
            'verbose': True,
        }

        # 下载视频和字幕
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

        # 检查视频和字幕是否下载成功
        video_file_exists = any(
            f.startswith(video_id) and f.endswith(('.mp4', '.mkv', '.webm'))
            for f in os.listdir(output_dir)
        )
        subtitle_files = [
            f for f in os.listdir(output_dir)
            if f.startswith(video_id) and f.endswith('.vtt')
        ]
        subtitle_exists = len(subtitle_files) > 0

        if video_file_exists and subtitle_exists:
            log_successful_id(video_id, succeed_log_file)
            # print(f"✅ 成功下载 {video_id} 的视频和字幕到 {output_dir}")
        else:
            # 尝试使用 YouTubeTranscriptApi 获取转录
            transcript_exists = False
            transcript_file = f'{output_dir}/{video_id}_transcript.txt'
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages = subtitleslangs)
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    for entry in transcript:
                        f.write(f"{entry['start']} - {entry['text']}\n")
                transcript_exists = True
                # print(f"✅ 成功获取 {video_id} 的转录并保存到 {transcript_file}")
            except Exception as e:
                # print(f"⚠️ {video_id} 无可用字幕或转录: {e}")
                pass

            # 如果字幕 + 视频 + 转录都失败，则标记失败
            if not (video_file_exists and (subtitle_exists or transcript_exists)):
                log_failed_id(video_id, "Missing subtitles and transcript", failed_log_file)
                # print(f"❌ 失败: {video_id} 下载完成但缺少字幕和转录")
            else:
                log_successful_id(video_id, succeed_log_file)
                # print(f"✅ 成功: {video_id} 视频 + 转录（无字幕）")

        time.sleep(0.6)  # 添加延迟，避免触发 YouTube 访问限制

    except Exception as e:
        print(f"❌ 下载 {video_id} 失败: {e}")
        log_failed_id(video_id, f"Download failed: {e}", failed_log_file)


# 主程序
def main():
    txt_file = 'youtube-asl_youtube_asl_video_ids.txt'  # 输入的 ID 文件
    succeed_log_file = 'successful_ids.txt'  # 成功记录文件
    failed_log_file = 'failed_ids.txt'  # 失败记录文件
    if os.path.exists(failed_log_file):
        ## ranme a new one
        failed_log_file = f'failed_ids_{time.strftime("%Y%m%d%H%M%S")}.txt'


    video_ids = read_video_ids(txt_file)

    exist_ids = read_video_ids(succeed_log_file)

    video_ids = list(set(video_ids) - set(exist_ids))
    
    for i, video_id in enumerate(video_ids):
        print(f"{i+1}/{len(video_ids)}: {video_id}")
        download_video_and_subtitles(video_id, succeed_log_file=succeed_log_file, failed_log_file=failed_log_file)
        if i % 10 == 0:
            print("暂停 10 秒")
            time.sleep(10)

if __name__ == "__main__":
    main()