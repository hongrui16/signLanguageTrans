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
import torch.distributed as dist

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)
    

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping

class How2SignNaive(Dataset):
    def __init__(self, split, root_dir = None, num_frames_per_clip = 150, **kwargs):
        """
        Initialize the YouTube ASL dataset loader for pre-processed frames.
        Args:
            split (str): Dataset split, e.g., 'train', 'val', 'test'.
            root_dir (str, optional): Root directory where the dataset is stored. If None,
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.logger = kwargs.get('logger', None)
        self.debug = kwargs.get('debug', False)
        self.modality = kwargs.get('modality', 'pose')
        self.num_frames_per_clip = num_frames_per_clip
        self.frame_size = kwargs.get('img_size', (224, 224))
        
        if 'rgb' in self.modality:
            self.load_frame = True
        else:
            self.load_frame = False
        
        if root_dir is None:
            self.root_dir = '/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/'
        else:
            self.root_dir = root_dir
            
        self.split = split
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")
        
        self.split_dir = os.path.join(self.root_dir, self.split)
        
        self.clip_frame_dir = os.path.join(self.split_dir, 'frames')
        self.clip_anno_dir = os.path.join(self.split_dir, 'annos')
        
        self.ann_filepath_txt = os.path.join(self.split_dir, f'{split}_annos_filepath.txt')
        
        if not os.path.exists(self.ann_filepath_txt):
            annos_files = os.listdir(self.clip_anno_dir)
            annos_files = [os.path.join(self.clip_anno_dir, f) for f in annos_files if f.endswith('.json')]
            with open(self.ann_filepath_txt, 'w', encoding='utf-8') as f:
                for anno_file in annos_files:
                    f.write(anno_file + '\n')

        
        self.annos_filepaths = self.load_annos_filepath(self.ann_filepath_txt)
        # self.annos_info = self._find_load_clips_annos()
                
        self.hand_mapping = MediapipeKptsMapping.hand_keypoints_mapping
        self.face_mapping = MediapipeKptsMapping.face_keypoints_mapping
        self.body_mapping = MediapipeKptsMapping.body_keypoints_mapping

        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = [value for key, value in self.hand_mapping.items()]  # Map to MediaPipe hand landmarks (0–20)
        self.body_indices = [value for key, value in self.body_mapping.items()]  # Map to MediaPipe body landmarks (0–8)
        self.face_indices = [value for key, value in self.face_mapping.items()]
                
        
        if self.debug:
            max_num_clips = 1000 if 1000 < len(self.annos_filepaths) else len(self.annos_filepaths)
            self.annos_filepaths = self.annos_filepaths[:max_num_clips]

        if not self.logger is None:
            self.logger.info(f"Total clips: {len(self.annos_filepaths)}")
        else:
            print(f"Total clips: {len(self.annos_filepaths)}")

   
   

    def load_annos_filepath(self, anno_filepath_txt) -> List[Tuple[str, str]]:
        # read the annotation file paths from the text file
        annos_filepaths = []
        with open(anno_filepath_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not os.path.exists(line):
                    print(f"Annotation file {line} does not exist, skipping.")
                    continue
                annos_filepaths.append(line)
        return annos_filepaths

    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.annos_filepaths)

    def __getitem__(self, idx: int, retry_count=0) -> Tuple[torch.Tensor, str, dict]:
        # if self.debug:
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        #     if self.logger:
        #         self.logger.info(f"how2signNaive [Rank {rank}] fetching sample index {idx}")
        #     else:
        #         print(f"how2signNaive [Rank {rank}] fetching sample index {idx}")

        clip_json_filepath = self.annos_filepaths[idx]
        if not os.path.exists(clip_json_filepath):
            if self.logger:
                self.logger.error(f"Annotation file {clip_json_filepath} does not exist, skipping.")
            else:
                print(f"Annotation file {clip_json_filepath} does not exist, skipping.")
            return self.__getitem__(np.random.randint(0, len(self.annos_filepaths)-1), retry_count + 1)
        
        with open(clip_json_filepath, 'r', encoding='utf-8') as f:
            clip_anno_info = json.load(f)


        json_filename = os.path.basename(clip_json_filepath)
        frames_filename = json_filename.replace('_anno.json', '_frames.mp4')
        
        frames_filepath = os.path.join(self.clip_frame_dir, frames_filename)
        
        if not os.path.exists(frames_filepath):
            if self.logger:
                self.logger.error(f"Frames file {frames_filepath} does not exist for clip {json_filename}, skipping.")
            else:
                print(f"Frames file {frames_filepath} does not exist for clip {json_filename}, skipping.")
            ran_id = np.random.randint(0, len(self.annos_filepaths)-1)
            return self.__getitem__(ran_id, retry_count + 1)
        
        ### read all frames from the video file
        if self.load_frame:
            cap = cv2.VideoCapture(frames_filepath)
            if not cap.isOpened():
                if self.logger:
                    self.logger.error(f"Failed to open video file {frames_filepath} for clip {json_filename}.")
                else:
                    print(f"Failed to open video file {frames_filepath} for clip {json_filename}.")
                ran_id = np.random.randint(0, len(self.annos_filepaths)-1)
                return self.__getitem__(ran_id, retry_count + 1)

            all_frames = []
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, self.frame_size)
                all_frames.append(frame_rgb)
            cap.release()

        # Read text
        text = clip_anno_info['text']
        
        keyframes_dict = clip_anno_info['keyframes']
        
        frame_names = list(keyframes_dict.keys())

     

        if len(frame_names) > self.num_frames_per_clip:
            ## uniformly sample frames            
            frame_names = self.uniform_with_jitter_sorted(frame_names, self.num_frames_per_clip, jitter_ratio=0.4)
            
        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        for frame_name in frame_names:

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
            frames_tensor = torch.tensor(0, dtype=torch.float32)  # Placeholder if frames are not loaded
        
        
        ## bady_keypoints_all: (N, 9, 2)
        ## hand_keypoints_all: (N, 42, 2)  # right hand + left hand
        ## face_keypoints_all: (N, 18, 2
            
        middle_point_of_shoulders = (body_keypoints_all[:, 2] + body_keypoints_all[:, 5]) / 2.0 # shape (N, 2)
        
        valid_body_mask = body_keypoints_all > 0
        valid_hand_mask = hand_keypoints_all > 0
        valid_face_mask = face_keypoints_all > 0
        
        ## Normalize all keypoints to the middle point of shoulders
        hand_keypoints_all[:, :, :2] -= middle_point_of_shoulders[:, None, :]
        body_keypoints_all[:, :, :2] -= middle_point_of_shoulders[:, None, :]
        face_keypoints_all[:, :, :2] -= middle_point_of_shoulders[:, None, :]
        
        ## Set invalid keypoints to -1
        hand_keypoints_all[~valid_hand_mask] = -1
        body_keypoints_all[~valid_body_mask] = -1
        face_keypoints_all[~valid_face_mask] = -1
        
        # to tensor
        body_keypoints_all = torch.from_numpy(body_keypoints_all).float() # shape (N, 9, 2)
        hand_keypoints_all = torch.from_numpy(hand_keypoints_all).float()
        face_keypoints_all = torch.from_numpy(face_keypoints_all).float()

        # hand_keypoints_all[:, :21] -=  hand_keypoints_all[:, :1] # normalize to wrist
        # hand_keypoints_all[:, 21:] -=  hand_keypoints_all[:, 21:22] # normalize to wrist

        # Create keypoints dictionary
        keypoints_dict = {
            'hand': hand_keypoints_all, # shape (N, 42, 2)  # right hand + left hand
            'body': body_keypoints_all, # shape (N, 9, 2)
            'face': face_keypoints_all # shape (N, 18, 2)
        }

        return (frames_tensor, text, keypoints_dict)

    def uniform_with_jitter_sorted(self, frame_names, num_samples, jitter_ratio=0.4):
        total_frames = len(frame_names)
        if total_frames <= num_samples:
            return frame_names  # 不足就全取

        base_positions = np.linspace(0, total_frames - 1, num_samples)
        interval = (total_frames - 1) / (num_samples - 1)
        max_jitter = interval * jitter_ratio

        jitter = np.random.uniform(-max_jitter, max_jitter, size=num_samples)
        jittered_positions = np.clip(np.round(base_positions + jitter), 0, total_frames - 1).astype(int)

        # 强制递增，确保顺序不乱，重复帧可接受
        jittered_positions = np.sort(jittered_positions)

        return [frame_names[i] for i in jittered_positions]


if __name__ == '__main__':


    split = 'test'
    dataset = How2SignNaive(split, load_frame = True)
  
    
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

    hand_keypoints_all = keypoints_dict['hand'] # right hand + left hand
    body_keypoints_all = keypoints_dict['body']
    face_keypoints_all = keypoints_dict['face']
    print(f"Hand keypoints shape: {hand_keypoints_all.shape}")
    print(f"Body keypoints shape: {body_keypoints_all.shape}")
    print(f"Face keypoints shape: {face_keypoints_all.shape}")

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