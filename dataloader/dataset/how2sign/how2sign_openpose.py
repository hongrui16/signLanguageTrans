import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import json
import heapq
import sys


import json
import re
import pandas as pd


from utils.openpose_kpts_mapping import OpenposeKptsMapping

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)
    



def natural_sort_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]

def extract_frame_id(filename):
    match = re.search(r'_(\d+)_keypoints\.json$', filename)
    return int(match.group(1)) if match else -1

class How2SignOpenPose(Dataset):
    def __init__(self, sentence_csv_path, kpts_json_dir, video_dir, n_frames=15, debug=False, **kwargs):
        self.debug = debug
        self.sentence_csv = pd.read_csv(sentence_csv_path, sep='\t' if '\t' in open(sentence_csv_path).readline() else ',')
        
        pose_seq_len = kwargs.get('pose_seq_len', 90)  # Default for YouTubeASL
        num_frame_seq = kwargs.get('frame_seq_len', 30)  # Default for


        self.kpts_json_dir = kpts_json_dir
        self.video_dir = video_dir
        self.n_frames = n_frames
        
        ## filter out the sentences json folder that does not exist
        self.sentence_csv = self.sentence_csv[self.sentence_csv['SENTENCE_NAME'].apply(lambda x: os.path.exists(os.path.join(kpts_json_dir, x)))]
        self.sentence_csv = self.sentence_csv.reset_index(drop=True)

        self.sentence_csv = self.sentence_csv.sort_values(by="SENTENCE_NAME", key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)

        self.hand_mapping = OpenposeKptsMapping.hand_keypoints_mapping
        self.face_mapping = OpenposeKptsMapping.face_keypoints_mapping
        self.body_mapping = OpenposeKptsMapping.body_keypoints_mapping

        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = [value for key, value in self.hand_mapping.items()]  # Map to MediaPipe hand landmarks (0–20)
        self.body_indices = [value for key, value in self.body_mapping.items()]  # Map to MediaPipe body landmarks (0–8)
        self.face_indices = [value for key, value in self.face_mapping.items()]

        # Define expected keypoint counts
        self.num_body_kpts = len(self.body_indices)  # e.g., 9
        self.num_hand_kpts = len(self.hand_indices)  # e.g., 21
        self.num_face_kpts = len(self.face_indices)  # e.g., 68

        self.num_clips = len(self.sentence_csv)



    def __len__(self):
        return len(self.sentence_csv)

    def __getitem__(self, idx):
        row = self.sentence_csv.iloc[idx]
        sentence_name = row['SENTENCE_NAME']
        sentence_text = row['SENTENCE']
        video_name = row['VIDEO_NAME']

        sentence_kpts_json_dir = os.path.join(self.kpts_json_dir, sentence_name)
        
        if not os.path.exists(sentence_kpts_json_dir):
            random_id = np.random.randint(0, self.num_clips)
            self.__getitem__(random_id)
        
        
        
        frame_width = 1280
        frame_height = 720
        
        all_jsons = [
            f for f in os.listdir(sentence_kpts_json_dir)
            if f.endswith("_keypoints.json") and f.startswith(sentence_name)
        ]
        all_jsons = sorted(all_jsons, key=extract_frame_id)

        # 均匀采样 n_frames（不足则补）
        if len(all_jsons) >= self.n_frames:
            indices = np.linspace(0, len(all_jsons) - 1, self.n_frames, dtype=int)
            selected_jsons = [all_jsons[i] for i in indices]
        else:
            selected_jsons = all_jsons + [all_jsons[-1]] * (self.n_frames - len(all_jsons))

        pose, lhand, rhand, face = [], [], [], []

        for json_file in selected_jsons:
            json_path = os.path.join(sentence_kpts_json_dir, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read {json_path}: {e}")
                pose.append(torch.zeros(self.num_body_kpts, 3))
                lhand.append(torch.zeros(self.num_hand_kpts, 3))
                rhand.append(torch.zeros(self.num_hand_kpts, 3))
                face.append(torch.zeros(self.num_face_kpts, 3))
                continue

            people = data.get('people', [])
            if not people:
                print(f"Warning: No people in {json_path}")
                pose.append(torch.zeros(self.num_body_kpts, 3))
                lhand.append(torch.zeros(self.num_hand_kpts,  3))
                rhand.append(torch.zeros(self.num_hand_kpts, 3))
                face.append(torch.zeros(self.num_face_kpts, 3))
                continue

            person = people[0]

            # Extract and map pose keypoints
            pose_kpts = person.get('pose_keypoints_2d', [])
            pose_kpts = np.array(pose_kpts).reshape(-1, 3)
            pose_kpts = pose_kpts[self.body_indices].astype(np.float32)
            # Normalize to [0, 1]
            pose_kpts[:, 0] /= frame_width
            pose_kpts[:, 1] /= frame_height
            pose.append(torch.tensor(pose_kpts, dtype=torch.float32))

            # Extract and map left hand keypoints
            lhand_kpts = person.get('hand_left_keypoints_2d', [])
            lhand_kpts = np.array(lhand_kpts).reshape(-1, 3)
            lhand_kpts = lhand_kpts[self.hand_indices].astype(np.float32)
            # Normalize to [0, 1]
            lhand_kpts[:, 0] /= frame_width
            lhand_kpts[:, 1] /= frame_height
            lhand.append(torch.tensor(lhand_kpts, dtype=torch.float32))
            
            # Extract and map right hand keypoints
            rhand_kpts = person.get('hand_right_keypoints_2d', [])
            rhand_kpts = np.array(rhand_kpts).reshape(-1, 3)
            rhand_kpts = rhand_kpts[self.hand_indices].astype(np.float32)
            # Normalize to [0, 1]
            rhand_kpts[:, 0] /= frame_width
            rhand_kpts[:, 1] /= frame_height
            rhand.append(torch.tensor(rhand_kpts, dtype=torch.float32))
            # Extract and map face keypoints
            face_kpts = person.get('face_keypoints_2d', [])
            face_kpts = np.array(face_kpts).reshape(-1, 3)
            face_kpts = face_kpts[self.face_indices].astype(np.float32)
            # Normalize to [0, 1]
            face_kpts[:, 0] /= frame_width
            face_kpts[:, 1] /= frame_height
            face.append(torch.tensor(face_kpts, dtype=torch.float32))


        rhand = torch.stack(rhand)
        lhand = torch.stack(lhand)
        face = torch.stack(face)
        pose = torch.stack(pose)
        
        hand = torch.cat((rhand, lhand), dim=1) # right hand + left hand

        pose = pose[:,:,:2]
        hand = hand[:,:,:2]
        face = face[:,:,:2]

        keypoints_dict = {
            "hand": hand, # right hand + left hand, size: (n_frames, 42, 2)
            "body": pose, # body, size: (n_frames, 9, 2)
            "face": face # face, size: (n_frames, 18, 2)
        }

        frames_tensor = torch.tensor(0)

        text = sentence_text

        return (frames_tensor, text, keypoints_dict)

            
            