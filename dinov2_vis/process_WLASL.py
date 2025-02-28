import cv2
import os, sys
import numpy as np
import json
from collections import Counter
import shutil
import urllib.request


def parse_json(file_path):
    video_info = {}
    video_info['gloss'] = []
    video_info['video_name'] = []
    video_info['url'] = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

            for key, value in data.items():
                video_info['video_name'].append(key)
                video_info['gloss'].append(value['gloss'])                
                video_info['url'].append(value['url'])
        return video_info
    
    except:
        print("Error reading file")
        return None
    
def get_top_glosses(video_info, top_n=15):
    # Count occurrences of each gloss
    gloss_counts = Counter(video_info['gloss'])
    top_glosses = gloss_counts.most_common(top_n)

    # Get video IDs for the top glosses
    top_videos = []
    for gloss, _ in top_glosses:
        for idx, gloss_name in enumerate(video_info['gloss']):
            if gloss_name == gloss:
                top_videos.append((video_info['video_name'][idx], gloss, video_info['url'][idx]))

    return top_videos

def download_video(url, video_filepath):
    try:
        urllib.request.urlretrieve(url, video_filepath)  # Use urllib to download the video
        print(f"Downloaded: {video_filepath}")
    except Exception as e:
        print(f"Failed to download video: {e}")



# Example usage
file_path = "/scratch/rhong5/dataset/signLanguage/WLASL/WLASL100/train/train.json"  # Replace with the path to your JSON file
video_info = parse_json(file_path)

video_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/raw_videos'
new_video_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/WLASL15_videos'
os.makedirs(new_video_dir, exist_ok=True)


print('Number of videos:', len(video_info['video_name']))

print('Number of glosses:', len(video_info['gloss']))

unique_glosses = list(set(video_info['gloss']))
print('Number of unique glosses:', len(unique_glosses))

# Get the top 15 glosses with the most videos
top_videos = get_top_glosses(video_info, top_n=15)
print("Top 15 glosses with corresponding video IDs:")
for video_id, gloss, url in top_videos:
    video_filepath = os.path.join(video_dir, video_id + '.mp4')
    new_video_filepath = os.path.join(new_video_dir, f'{gloss}_{video_id}.mp4')
    if os.path.exists(video_filepath):
        shutil.copy(video_filepath, new_video_filepath)
    else:
        # download video from url
        download_video(url, new_video_filepath)
