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

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)
    

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping

class YouTubeASLClip(Dataset):
    def __init__(self, clip_frame_dir: str, clip_anno_dir: str, num_frames_per_clip: int = 15, **kwargs):
        """
        Initialize the YouTube ASL dataset loader for pre-cropped clips.
        
        Args:
            clip_dir (str): Directory containing clip (.mp4) and transcript (.txt) files (e.g., 'clip_0.mp4', 'clip_0.txt').
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.clip_frame_dir = clip_frame_dir
        self.clip_anno_dir = clip_anno_dir
        self.num_frames_per_clip = num_frames_per_clip
        self.debug = kwargs.get('debug', False)
        self.logger = kwargs.get('logger', None)
        
        self.load_frame = kwargs.get('load_frame', False)
        
        self.frame_size = kwargs.get('frame_size', (224, 224))

        
        self.hand_mapping = MediapipeKptsMapping.hand_keypoints_mapping
        self.face_mapping = MediapipeKptsMapping.face_keypoints_mapping
        self.body_mapping = MediapipeKptsMapping.body_keypoints_mapping

        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = [value for key, value in self.hand_mapping.items()]  # Map to MediaPipe hand landmarks (0–20)
        self.body_indices = [value for key, value in self.body_mapping.items()]  # Map to MediaPipe body landmarks (0–8)
        self.face_indices = [value for key, value in self.face_mapping.items()]
        
        # self.annos_info = self._find_load_clips_annos()
        self.annos_files = self._find_clips_annos()
        
        if self.debug:
            max_num_clips = 1000 if 1000 < len(self.annos_files) else len(self.annos_files)
            self.annos_files = self.annos_files[:max_num_clips]

        if not self.logger is None:
            self.logger.info(f"Total clips: {len(self.annos_files)}")
        else:
            print(f"Total clips: {len(self.annos_files)}")

    def _find_load_clips_annos(self) -> List[Tuple[str, str]]:
        """
        Find matching clip anno (.json) 
        
        Returns:
            List of tuples json file paths
        """
        clip_files = sorted([f for f in os.listdir(self.clip_anno_dir) if f.endswith('.json')])
        clip_annos = []
        
        for clip_file in clip_files:
            clip_path = os.path.join(self.clip_anno_dir, clip_file)
            with open(clip_path, 'r', encoding='utf-8') as f:
                clip_anno = json.load(f)
                clip_annos.append(clip_anno)
        return clip_annos

    def _find_clips_annos(self) -> List[Tuple[str, str]]:
        """
        Find matching clip anno (.json) 
        
        Returns:
            List of tuples json file paths
        """
        clip_files = sorted([f for f in os.listdir(self.clip_anno_dir) if f.endswith('.json')])
        return clip_files
        # clip_annos = []
        
        # for clip_file in clip_files:
        #     clip_path = os.path.join(self.clip_anno_dir, clip_file)
        #     with open(clip_path, 'r', encoding='utf-8') as f:
        #         clip_anno = json.load(f)
        #         clip_annos.append(clip_anno)
        # return clip_annos


    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.annos_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, dict]:
        """
        Get a specific clip from the dataset.
        
        Args:
            idx (int): Index of the clip.
        
        Returns:
            Tuple of (frames_tensor, text, keypoints_dict) where:
            - frames_tensor: Tensor of shape (N, 3, 224, 224)
            - text: String caption for the clip
            - keypoints_dict: Dict with 'hand', 'body', 'face' keys, each a list of (x, y, confidence)
        """
        clip_file = self.annos_files[idx]
        clip_path = os.path.join(self.clip_anno_dir, clip_file)
        with open(clip_path, 'r', encoding='utf-8') as f:
            clip_anno_info = json.load(f)

        
        # Read text
        text = clip_anno_info['text']
        
        keyframes_dict = clip_anno_info['keyframes']
        
        frame_names = list(keyframes_dict.keys())

        if self.load_frame:
            frame_names = [f for f in frame_names if os.path.exists(os.path.join(self.clip_frame_dir, f))]

            

        if len(frame_names) > self.num_frames_per_clip:
            frame_ids_offest = []
            start = 0
            for frame_name in frame_names:
                frame_id = frame_name.split('_')[-1].split('.')[0]
                int_frame_id = int(frame_id)
                frame_ids_offest.append(int_frame_id - start)
                start = int_frame_id            
            top_n_indices = heapq.nlargest(self.num_frames_per_clip, range(len(frame_ids_offest)), key=lambda i: frame_ids_offest[i])
            frame_names = [frame_names[i] for i in top_n_indices]

        all_frames = []
        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        for frame_name in frame_names:
            if self.load_frame:
                frame_path = os.path.join(self.clip_frame_dir, frame_name)
                frame_bgr = cv2.imread(frame_path)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, self.frame_size)
                all_frames.append(frame_rgb)

            frame_keypoints = keyframes_dict[frame_name]
            hand_keypoints = frame_keypoints['hand']
            body_keypoints = frame_keypoints['body']
            face_keypoints = frame_keypoints['face']
            hand_keypoints_all.append(hand_keypoints)
            body_keypoints_all.append(body_keypoints)
            face_keypoints_all.append(face_keypoints)
            
        
        body_keypoints_all = np.array(body_keypoints_all, dtype=np.float32)
        hand_keypoints_all = np.array(hand_keypoints_all, dtype=np.float32)
        face_keypoints_all = np.array(face_keypoints_all, dtype=np.float32)


        if len(frame_names) < self.num_frames_per_clip:
            body_keypoints_all = np.concatenate((body_keypoints_all, np.zeros((self.num_frames_per_clip - len(body_keypoints_all), len(self.body_indices), 2), dtype=np.float32) - 1), axis=0)
            hand_keypoints_all = np.concatenate((hand_keypoints_all, np.zeros((self.num_frames_per_clip - len(hand_keypoints_all), 2*len(self.hand_indices), 2), dtype=np.float32) - 1), axis=0)
            face_keypoints_all = np.concatenate((face_keypoints_all, np.zeros((self.num_frames_per_clip - len(face_keypoints_all), len(self.face_indices), 2), dtype=np.float32) - 1), axis=0)


        # Stack frames into a tensor
        if self.load_frame:
            all_frames = np.array(all_frames, dtype=np.uint8)
            if all_frames.shape[0] < self.num_frames_per_clip:                    
                pad_frames = np.zeros((self.num_frames_per_clip - len(all_frames), self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                all_frames = np.concatenate((all_frames, pad_frames), axis=0)
            frames_tensor = torch.from_numpy(all_frames).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, 224, 224)
        else:
            frames_tensor = 0
            
        # to tensor
        body_keypoints_all = torch.from_numpy(body_keypoints_all).float()
        hand_keypoints_all = torch.from_numpy(hand_keypoints_all).float()
        face_keypoints_all = torch.from_numpy(face_keypoints_all).float()


        # Create keypoints dictionary
        keypoints_dict = {
            'hand': hand_keypoints_all,
            'body': body_keypoints_all,
            'face': face_keypoints_all
        }

        return (frames_tensor, text, keypoints_dict)


if __name__ == '__main__':

    clip_anno_dir = '/scratch/rhong5/dataset/youtubeASL_anno'
    clip_frame_dir = '/scratch/rhong5/dataset/youtubeASL_frames'


    dataset = YouTubeASLClip(clip_frame_dir, clip_anno_dir, load_frame=True)
  
    
    # get one sample and draw keypoints on the image

    id = np.random.randint(0, len(dataset)-1)
    frames_tensor, text, keypoints_dict = dataset[id]

    print(f"Text: {text}")
    print(f"Frames tensor shape: {frames_tensor.shape}")
    # print(f"Keypoints dict: {keypoints_dict}")
    print(f"Hand keypoints shape: {keypoints_dict['hand'].shape}")
    print(f"Body keypoints shape: {keypoints_dict['body'].shape}")
    print(f"Face keypoints shape: {keypoints_dict['face'].shape}")
    # Draw keypoints on the first frame
    frame = frames_tensor[0].permute(1, 2, 0).numpy() * 255
    frame = frame.astype(np.uint8)

    h, w, _ = frame.shape
    # Draw keypoints
    hand_keypoints = keypoints_dict['hand'][0].numpy()
    body_keypoints = keypoints_dict['body'][0].numpy()
    face_keypoints = keypoints_dict['face'][0].numpy()
    for i in range(len(hand_keypoints)):
        x, y = hand_keypoints[i]
        if x != -1 and y != -1:
            cv2.circle(frame, (int(x*w), int(y*h)), 1, (0, 255, 0), -1)
    for i in range(len(body_keypoints)):
        x, y = body_keypoints[i]
        if x != -1 and y != -1:
            cv2.circle(frame, (int(x*w), int(y*h)), 1, (255, 0, 0), -1)
    for i in range(len(face_keypoints)):
        x, y = face_keypoints[i]
        if x != -1 and y != -1:
            cv2.circle(frame, (int(x*w), int(y*h)), 1, (0, 0, 255), -1)

    cv2.imwrite('keypoints.jpg', frame[:,:,::-1])