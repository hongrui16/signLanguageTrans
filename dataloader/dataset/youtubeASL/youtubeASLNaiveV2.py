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

class YoutubeASLNaiveV2(Dataset):
    def __init__(self, split, root_dir = None, pose_seq_len = 150, frame_seq_len = 60, **kwargs):
        """
        Initialize the YouTube ASL dataset loader for pre-processed frames.
        Args:
            split (str): Dataset split, e.g., 'train', 'val', 'test'.
            root_dir (str, optional): Root directory where the dataset is stored. If None,
            pose_seq_len (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.logger = kwargs.get('logger', None)
        self.debug = kwargs.get('debug', False)
        self.modality = kwargs.get('modality', 'pose')
        self.pose_seq_len = pose_seq_len
        self.frame_seq_len = frame_seq_len
        self.frame_size = kwargs.get('img_size', (224, 224))
        self.delete_blury_frames = kwargs.get('delete_blury_frames', False)  # Default for YouTubeASL
        self.use_mini_dataset = kwargs.get('use_mini_dataset', False)  # Default for debugging
        self.xD_pose = kwargs.get('xD_pose', '2D')  # Default for 2D pose
        
        if 'rgb' in self.modality:
            self.load_frame = True
        else:
            self.load_frame = False
            
        if 'pose' in self.modality:
            self.load_pose = True
        else:
            self.load_pose = False
        
        
        if root_dir is None:
            self.root_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722'
        else:
            self.root_dir = root_dir
            
        self.split = split
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")
        
        if self.split == 'val':
            self.split = 'test'
            
        # self.split_dir = os.path.join(self.root_dir, self.split)
        # self.split_dir = self.root_dir
        
        self.clip_frame_dir = os.path.join(self.root_dir, 'frames')
        self.clip_anno_dir = os.path.join(self.root_dir, 'annos')
        
        if self.use_mini_dataset:
            self.ann_filepath_txt = os.path.join(self.root_dir, f'mini_{split}_annos_filepath.txt')
        else:
            self.ann_filepath_txt = os.path.join(self.root_dir, f'{split}_annos_filepath.txt')
        
        if not os.path.exists(self.ann_filepath_txt):
            annos_files = os.listdir(self.clip_anno_dir)
            annos_files = [os.path.join(self.clip_anno_dir, f) for f in annos_files if f.endswith('.json')]
            if self.use_mini_dataset:
                annos_files = annos_files[:len(annos_files)//2]  # Limit to 1000 samples for debugging
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
        
        
        self.num_hand_kpts = len(self.hand_indices)*2  # 42 for both hands
        self.num_body_kpts = len(self.body_indices)  # 9 for body
        self.num_face_kpts = len(self.face_indices)  # 18 for face
                
        
        if self.debug:
            max_num_clips = 1000 if 1000 < len(self.annos_filepaths) else len(self.annos_filepaths)
            self.annos_filepaths = self.annos_filepaths[:max_num_clips]

        if not self.logger is None:
            self.logger.info(f"{self.split}, Total clips: {len(self.annos_filepaths)}")
        else:
            print(f"{self.split}, Total clips: {len(self.annos_filepaths)}")

   
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
                all_frames.append(frame_rgb)
            cap.release()
            
            all_frames = np.array(all_frames, dtype=np.uint8) # (num_seq, 224, 224, 3)
            # print('all_frames shape:', all_frames.shape)

        # Read text
        text = clip_anno_info['text']
        
        keyframes_dict = clip_anno_info['keyframes']
        
        frame_names = list(keyframes_dict.keys())

        width = clip_anno_info['width']
        height = clip_anno_info['height']
        body_bbox = clip_anno_info['body_bbox']

        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        new_frames_names = []
        for frame_name in frame_names:
            frame_keypoints = keyframes_dict[frame_name]
            hand_keypoints = frame_keypoints['hand']
            np_hand_keypoints = np.array(hand_keypoints, dtype=np.float32)
            if self.delete_blury_frames and (np_hand_keypoints < 0).any():
                # Skip frames with no hand keypoints
                ### if the chance is larger than 0.6, keep the frame
                if np.random.rand() <= 0.2:                
                    continue
                
            new_frames_names.append(frame_name)
            if self.load_pose:
                body_keypoints = frame_keypoints['body']
                face_keypoints = frame_keypoints['face']
                
                hand_keypoints_all.append(hand_keypoints)
                body_keypoints_all.append(body_keypoints)
                face_keypoints_all.append(face_keypoints)
                
            
        if len(new_frames_names) == 0:
            # If no valid frames, retry with a different index
            ran_id = np.random.randint(0, len(self.annos_filepaths)-1)
            self.logger.warning(f"No valid frames found for clip {json_filename}, retrying with random index {ran_id}.")
            return self.__getitem__(ran_id, retry_count + 1)

            
        frame_names = new_frames_names
        
        ## uniformly sample frames            
        
        
        if self.load_frame and self.load_pose:
            seq_len = min(self.frame_seq_len, self.pose_seq_len)
        elif self.load_frame:
            seq_len = self.frame_seq_len
        elif self.load_pose:
            seq_len = self.pose_seq_len
            
        selected_seq_indices, selected_seq_names = self.uniform_with_jitter_sorted(frame_names, seq_len, jitter_ratio=0.4)
        # print(f"Selected sequence indices: {len(selected_seq_indices)}")
        
        if self.load_pose:
            selected_pose_seq_indices = selected_seq_indices
            # selected_pose_seq_indices, selected_pose_seq_names = self.uniform_with_jitter_sorted(frame_names, self.pose_seq_len, jitter_ratio=0.4)
            
            body_keypoints_all = np.array(body_keypoints_all, dtype=np.float32)
            hand_keypoints_all = np.array(hand_keypoints_all, dtype=np.float32)
            face_keypoints_all = np.array(face_keypoints_all, dtype=np.float32)
            
            ### get the bbox of the body
            all_keypoints = np.concatenate((hand_keypoints_all, body_keypoints_all, face_keypoints_all), axis=1)
            all_keypoints[:, :, 0] *= width
            all_keypoints[:, :, 1] *= height
            # all_keypoints = all_keypoints.astype(int)
                        
            xmin, ymin, xmax, ymax = body_bbox
            
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            all_keypoints[:, :, 0] -= xmin
            all_keypoints[:, :, 1] -= ymin
            
            # all_keypoints = all_keypoints.astype(np.float32)  # Convert all keypoints to float32
            # all_keypoints[:, :, 0] /= new_width
            # all_keypoints[:, :, 1] /= new_height
            
            ## to float32
            hand_keypoints_all = all_keypoints[:, :self.num_hand_kpts].astype(np.float32)
            body_keypoints_all = all_keypoints[:, self.num_hand_kpts: self.num_hand_kpts + self.num_body_kpts].astype(np.float32)
            face_keypoints_all = all_keypoints[:, self.num_hand_kpts + self.num_body_kpts:,].astype(np.float32)
        
                    
            hand_keypoints_all = hand_keypoints_all[selected_pose_seq_indices]
            body_keypoints_all = body_keypoints_all[selected_pose_seq_indices]
            face_keypoints_all = face_keypoints_all[selected_pose_seq_indices]
            
            
                
            body_keypoints_all[:,:,0] /= bbox_width
            body_keypoints_all[:,:,1] /= bbox_height
            
            hand_keypoints_all[:,:,0] /= bbox_width
            hand_keypoints_all[:,:,1] /= bbox_height
            
            face_keypoints_all[:,:,0] /= bbox_width
            face_keypoints_all[:,:,1] /= bbox_height

            
            valid_body_mask = body_keypoints_all > 0
            valid_hand_mask = hand_keypoints_all > 0
            valid_face_mask = face_keypoints_all > 0
                        
            ## Set invalid keypoints to -1
            hand_keypoints_all[~valid_hand_mask] = 0
            body_keypoints_all[~valid_body_mask] = 0
            face_keypoints_all[~valid_face_mask] = 0
            
            num_frames = hand_keypoints_all.shape[0]

            if num_frames < self.pose_seq_len:
                pad_length = self.pose_seq_len - num_frames
                padded_body_kpts = np.zeros((pad_length, self.num_body_kpts, 3), dtype=np.float32) -1
                padded_hand_kpts = np.zeros((pad_length, self.num_hand_kpts, 3), dtype=np.float32) -1
                padded_face_kpts = np.zeros((pad_length, self.num_face_kpts, 3), dtype=np.float32) -1
                body_keypoints_all = np.concatenate((body_keypoints_all, padded_body_kpts), axis=0)
                hand_keypoints_all = np.concatenate((hand_keypoints_all, padded_hand_kpts), axis=0)                                                
                face_keypoints_all = np.concatenate((face_keypoints_all, padded_face_kpts), axis=0)

            valid_pose_seq_mask = np.zeros((hand_keypoints_all.shape[0],), dtype=bool)  # Assume all pose frames are valid initially
            valid_pose_seq_mask[:num_frames] = True  # Set valid frames to True
                
            # to tensor
            body_keypoints_all = torch.from_numpy(body_keypoints_all).float() # shape (N, 9, 2)
            hand_keypoints_all = torch.from_numpy(hand_keypoints_all).float()
            face_keypoints_all = torch.from_numpy(face_keypoints_all).float()
        else:
            body_keypoints_all = torch.tensor(0, dtype=torch.float32)
            hand_keypoints_all = torch.tensor(0, dtype=torch.float32)
            face_keypoints_all = torch.tensor(0, dtype=torch.float32)
            valid_pose_seq_mask = np.zeros((1,), dtype=bool)  # No valid pose frames if not loaded
            
            
        if self.load_frame:
            xmin, ymin, xmax, ymax = body_bbox
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            # selected_frame_seq_indices, selected_frame_seq_names = self.uniform_with_jitter_sorted(frame_names, self.frame_seq_len, jitter_ratio=0.4)
            selected_frame_seq_indices = selected_seq_indices
            
            all_frames = all_frames[selected_frame_seq_indices]
            # print(f"all_frames shape: {all_frames.shape}") #(num_seq, 224, 224, 3)
            # print('xmin, xmax, ymin, ymax:', xmin, xmax, ymin, ymax) ## 0 1225 0 719
            all_frames = all_frames[:, ymin:ymax, xmin:xmax]

            frames_tensor = torch.from_numpy(all_frames).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, 224, 224)         
            ## resize frames to (224, 224)
            frames_tensor = torch.nn.functional.interpolate(frames_tensor, size=self.frame_size, mode='bilinear', align_corners=False)
            num_frames = frames_tensor.shape[0]
            if num_frames < self.frame_seq_len:            
                pad_length = self.frame_seq_len - num_frames
                pad_frames = torch.zeros((pad_length, 3, self.frame_size[0], self.frame_size[1]), dtype=torch.float32)
                # print('all_frames shape:', all_frames.shape)
                # print('pad_frames shape:', pad_frames.shape)
                frames_tensor = torch.cat((frames_tensor, pad_frames), axis=0)
            valid_frame_seq_mask = np.zeros((frames_tensor.shape[0],), dtype=bool)  # Assume all frame sequences are valid initially
            valid_frame_seq_mask[:num_frames] = True  # Set valid frames to True
        else:
            frames_tensor = torch.tensor(0, dtype=torch.float32)  # Placeholder if frames are not loaded
            valid_frame_seq_mask = np.zeros((1,), dtype=bool)  # No valid frames if not loaded
        
        
        ## bady_keypoints_all: (N, 9, 2)
        ## hand_keypoints_all: (N, 42, 2)  # right hand + left hand
        ## face_keypoints_all: (N, 18, 2
            
        # Create keypoints dictionary
        keypoints_dict = {
            'hand': hand_keypoints_all, # shape (N, 42, 2)  # right hand + left hand
            'body': body_keypoints_all, # shape (N, 9, 2)
            'face': face_keypoints_all,  # shape (N, 18, 2)
            'valid_frame_seq_mask': valid_frame_seq_mask,  # shape (N,)
            'valid_pose_seq_mask': valid_pose_seq_mask,  # shape (N,)
        }

        return (frames_tensor, text, keypoints_dict)

    def uniform_with_jitter_sorted(self, frame_names, num_samples, jitter_ratio=0.4):
        total_frames = len(frame_names)
        if total_frames <= num_samples:
            return np.arange(total_frames), frame_names

        base_positions = np.linspace(0, total_frames - 1, num_samples)
        interval = (total_frames - 1) / (num_samples - 1)
        max_jitter = interval * jitter_ratio

        jitter = np.random.uniform(-max_jitter, max_jitter, size=num_samples)
        jittered_positions = np.clip(np.round(base_positions + jitter), 0, total_frames - 1).astype(int)

        # 强制递增，确保顺序不乱，重复帧可接受
        jittered_positions = np.sort(jittered_positions)
        
        jiltered_frame_names = [frame_names[i] for i in jittered_positions]

        return jittered_positions, jiltered_frame_names


if __name__ == '__main__':


    split = 'test'
    pose_seq_len = 90
    frame_seq_len = 90
    modility = 'pose_rgb'  # 'pose', 'rgb', 'pose_rgb'
    dataset = YoutubeASLNaiveV2(split, 
                            pose_seq_len=pose_seq_len, 
                            frame_seq_len=frame_seq_len, 
                            modality=modility, 
                            img_size=(224, 224), 
                            )
  
    
    # get one sample and draw keypoints on the image
    cnt = 0
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    while cnt <= 20:
        cnt += 1
        id = np.random.randint(0, len(dataset)-1)
        frames_tensor, text, keypoints_dict = dataset[id]

        # print(f"Text: {text}")
        print(f"Frames tensor shape: {frames_tensor.shape}")
        # # print(f"Keypoints dict: {keypoints_dict}")
        print(f"Hand keypoints shape: {keypoints_dict['hand'].shape}")
        print(f"Body keypoints shape: {keypoints_dict['body'].shape}")
        print(f"Face keypoints shape: {keypoints_dict['face'].shape}")
        # Draw keypoints on the first frame
        frame = frames_tensor[0].permute(1, 2, 0).numpy() * 255
        frame = frame.astype(np.uint8)
        frame = frame.copy()  # Create a copy to draw on
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape

        hand_keypoints_all = keypoints_dict['hand'] # right hand + left hand
        body_keypoints_all = keypoints_dict['body']
        face_keypoints_all = keypoints_dict['face']
        # print(f"Hand keypoints shape: {hand_keypoints_all.shape}")
        # print(f"Body keypoints shape: {body_keypoints_all.shape}")
        # print(f"Face keypoints shape: {face_keypoints_all.shape}")

        # Draw keypoints
        hand_keypoints = keypoints_dict['hand'][0].numpy()
        body_keypoints = keypoints_dict['body'][0].numpy()
        face_keypoints = keypoints_dict['face'][0].numpy()
        for i in range(len(hand_keypoints)):
            x, y, z = hand_keypoints[i]
            if x != -1 and y != -1:
                cv2.circle(frame, (int(x*w), int(y*h)), 1, (0, 255, 0), -1)
        for i in range(len(body_keypoints)):
            x, y, z = body_keypoints[i]
            if x != -1 and y != -1:
                cv2.circle(frame, (int(x*w), int(y*h)), 1, (255, 0, 0), -1)
        for i in range(len(face_keypoints)):
            x, y, z = face_keypoints[i]
            if x != -1 and y != -1:
                cv2.circle(frame, (int(x*w), int(y*h)), 1, (0, 0, 255), -1)

        cv2.imwrite(f'{temp_dir}/keypoints_{cnt}.jpg', frame)
        print()