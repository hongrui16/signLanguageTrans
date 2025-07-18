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

class YouTubeASLFramesNaive(Dataset):
    def __init__(self, split = 'train', ann_filepath_txt = None, **kwargs):
        """
        Initialize the YouTube ASL dataset loader for pre-processed frames.
        
        Args:
            clip_frame_dir (str): Directory containing frames (.jpg)
            clip_anno_dir (str): Directory containing annotations (.json).
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        if ann_filepath_txt is None:
            if split == 'train':
                self.ann_filepath_txt = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno_all_filepaths.txt'
            elif split == 'test':
                self.ann_filepath_txt = '/projects/kosecka/hongrui/dataset/youtubeASL/fake_test_filepaths.txt'
            else:
                raise ValueError(f"Unsupported split: {split}. Supported splits are 'train' and 'test'.")
                
        else:
            self.ann_filepath_txt = ann_filepath_txt

        self.clip_frame_dir = 'youtubeASL_frames'
        self.clip_anno_dir = 'youtubeASL_anno'
        
        pose_seq_len = kwargs.get('pose_seq_len', 90)  # Default for YouTubeASL
        num_frame_seq = kwargs.get('frame_seq_len', 30)  # Default for

        self.debug = kwargs.get('debug', False)
        self.logger = kwargs.get('logger', None)
        self.modality = kwargs.get('modality', 'pose')  # Default to video modality
        # assert self.modality in ['pose', 'rgb', 'rgb_pose'], f"Unsupported modality: {self.modality}"
        
        self.load_frame = True if 'rgb' in self.modality else False
        
        self.frame_size = kwargs.get('img_size', (224, 224))

        
        self.hand_mapping = MediapipeKptsMapping.hand_keypoints_mapping
        self.face_mapping = MediapipeKptsMapping.face_keypoints_mapping
        self.body_mapping = MediapipeKptsMapping.body_keypoints_mapping

        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = [value for key, value in self.hand_mapping.items()]  # Map to MediaPipe hand landmarks (0–20)
        self.body_indices = [value for key, value in self.body_mapping.items()]  # Map to MediaPipe body landmarks (0–8)
        self.face_indices = [value for key, value in self.face_mapping.items()]
        
        # self.annos_info = self._find_load_clips_annos()
        self.annos_files = self.load_annos_filepath()
        
            
        print(f"Loaded {len(self.annos_files)} annotation files from {self.ann_filepath_txt}")
        
        if self.debug:
            max_num_clips = 1000 if 1000 < len(self.annos_files) else len(self.annos_files)
            self.annos_files = self.annos_files[:max_num_clips]

        if not self.logger is None:
            self.logger.info(f"Total clips: {len(self.annos_files)}")
        else:
            print(f"Total clips: {len(self.annos_files)}")

   
   

    def load_annos_filepath(self) -> List[Tuple[str, str]]:
        # read the annotation file paths from the text file
        annos_files = []
        with open(self.ann_filepath_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not os.path.exists(line):
                    print(f"Annotation file {line} does not exist, skipping.")
                    continue
                annos_files.append(line)
        return annos_files

    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.annos_files)

    def __getitem__(self, idx: int, retry_count=0) -> Tuple[torch.Tensor, str, dict]:
        if self.debug:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if self.logger:
                self.logger.info(f"[Rank {rank}] fetching sample index {idx}")
            else:
                print(f"[Rank {rank}] fetching sample index {idx}")

        clip_json_file = self.annos_files[idx]
        try:
            with open(clip_json_file, 'r', encoding='utf-8') as f:
                clip_anno_info = json.load(f)
        except json.JSONDecodeError as e:
            if retry_count >= 5:
                raise RuntimeError(f"Too many corrupt JSON retries starting from index {idx}")
            if self.logger:
                self.logger.error(f"Error decoding JSON from {clip_json_file}: {e}")
            else:
                print(f"Error decoding JSON from {clip_json_file}: {e}")
            ran_id = np.random.randint(0, len(self.annos_files)-1)
            return self.__getitem__(ran_id, retry_count + 1)

        json_filename = os.path.basename(clip_json_file)
        json_dir = os.path.dirname(clip_json_file)

        frame_dir = json_dir.replace(self.clip_anno_dir, self.clip_frame_dir)

        # Read text
        text = clip_anno_info['text']
        
        keyframes_dict = clip_anno_info['keyframes']
        
        frame_names = list(keyframes_dict.keys())

     

        if len(frame_names) > self.num_frames_per_clip:
            ## uniformly sample frames
            frame_indices = np.linspace(0, len(frame_names) - 1, self.num_frames_per_clip).astype(int)
            frame_names = [frame_names[i] for i in frame_indices]

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


if __name__ == '__main__':



    dataset = YouTubeASLFramesNaive(load_frame = True)
  
    
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