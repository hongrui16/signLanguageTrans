import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import torch.nn.functional as F
import webvtt
import sys
from ultralytics import YOLO
import random
import json

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)


import functools
print = functools.partial(print, flush=True)


from utils.mediapipe_kpts_mapping import MediapipeKptsMapping




class MediapipePose:
    def __init__(self, split = 'train', debug = False):
        """
        Initialize the YouTube ASL dataset loader for a single video and VTT files.
        
        Args:
            video_dir (str): Directory containing the video ('video.mp4') and transcript (*.vtt) files.
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.root_dir = '/projects/kosecka/hongrui/dataset/how2sign/how2sign_pro_0714'
        self.video_dir = f'{self.root_dir}/{split}/frames'
        self.anno_dir = f'{self.root_dir}/{split}/annos_2D'
        self.new_anno_dir = f'{self.root_dir}/{split}/annos'
        
        os.makedirs(self.new_anno_dir, exist_ok=True)


        self.debug = debug

        self.keypoints_threshold = 0.15  # Confidence threshold for keypoint detection

        filepath_txt = os.path.join(self.root_dir, split , f'{split}_annos_filepath.txt')

        ### read file paths from txt file, every line is a video file path
        with open(filepath_txt, 'r') as f:
            self.annno_paths = [line.strip() for line in f.readlines()]

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        self.hand_mapping = MediapipeKptsMapping.hand_keypoints_mapping
        self.face_mapping = MediapipeKptsMapping.face_keypoints_mapping
        self.body_mapping = MediapipeKptsMapping.body_keypoints_mapping

        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = [value for key, value in self.hand_mapping.items()]  # Map to MediaPipe hand landmarks (0–20)
        self.body_indices = [value for key, value in self.body_mapping.items()]  # Map to MediaPipe body landmarks (0–8)
        self.face_indices = [value for key, value in self.face_mapping.items()]
    


    def _detect_3d_keypoints(self, frame_rgb: np.ndarray, det_hand_kpts: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect body, face, and hand 3D keypoints using MediaPipe.

        Args:
            frame_rgb (np.ndarray): RGB frame of shape (H, W, 3).

        Returns:
            Tuple of hand_keypoints, body_keypoints, face_keypoints.
            Each is an np.ndarray of shape (N, 3) with (x, y, z). Missing points are (-1.0, -1.0, -1.0).
        """
        results = self.holistic.process(frame_rgb)

        # --- Hand ---
        left_hand_keypoints = [(-1.0, -1.0, -1.0)] * len(self.hand_indices)
        right_hand_keypoints = [(-1.0, -1.0, -1.0)] * len(self.hand_indices)

        if det_hand_kpts and results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                if idx in self.hand_indices:
                    pos = self.hand_indices.index(idx)
                    x, y, z = landmark.x, landmark.y, landmark.z
                    left_hand_keypoints[pos] = (x, y, z)

        if det_hand_kpts and results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                if idx in self.hand_indices:
                    pos = self.hand_indices.index(idx)
                    x, y, z = landmark.x, landmark.y, landmark.z
                    right_hand_keypoints[pos] = (x, y, z)

        hand_keypoints = right_hand_keypoints + left_hand_keypoints

        # --- Pose (Body) ---
        body_keypoints = [(-1.0, -1.0, -1.0)] * len(self.body_indices)
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in self.body_indices:
                    pos = self.body_indices.index(idx)
                    x, y, z = landmark.x, landmark.y, landmark.z
                    body_keypoints[pos] = (x, y, z)

        # --- Face ---
        face_keypoints = [(-1.0, -1.0, -1.0)] * len(self.face_indices)
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                if idx in self.face_indices:
                    pos = self.face_indices.index(idx)
                    x, y, z = landmark.x, landmark.y, landmark.z
                    face_keypoints[pos] = (x, y, z)

        hand_keypoints = np.array(hand_keypoints, dtype=np.float32)
        body_keypoints = np.array(body_keypoints, dtype=np.float32)
        face_keypoints = np.array(face_keypoints, dtype=np.float32)

        return hand_keypoints, body_keypoints, face_keypoints



    def process_videos(self, video_path: str, clip_info_dict, print_flag: bool = False):

        
        body_bbox = clip_info_dict['body_bbox']
        height = clip_info_dict['height']
        width = clip_info_dict['width']
        height = int(height)
        width = int(width)
        

        # frame_info_dict['frame_name'] = frame_name
        formated_frame_ids = clip_info_dict['keyframes'].keys()
        
        cap = cv2.VideoCapture(video_path)

        for frame_index, formated_frame_id in enumerate(formated_frame_ids):      
            # if print_flag:
            #     print(f"Processing frame {frame_index + 1}/{len(formated_frame_ids)}: {formated_frame_id}")
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {frame_index} from video {video_path}. Skipping.")
                continue
            
            frame_info_dict = clip_info_dict['keyframes'][formated_frame_id]
            
            body_xmin, body_ymin, body_xmax, body_ymax = body_bbox
            body_xmin = int(body_xmin)
            body_ymin = int(body_ymin)
            body_xmax = int(body_xmax)
            body_ymax = int(body_ymax)

            ori_h, ori_w = frame.shape[:2]
            assert ori_h == height and ori_w == width, f"Frame size {ori_h}x{ori_w} does not match expected size {height}x{width}"
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print('body_bbox:', body_bbox)
            
            body_rgb = frame_rgb[body_ymin:body_ymax, body_xmin:body_xmax].copy()
            body_h, body_w = body_rgb.shape[:2]
            
            # Detect hands using MediaPipe
            hand_keypoints, body_keypoints, face_keypoints = self._detect_3d_keypoints(body_rgb)
            
            invalid_hand_mask = (hand_keypoints[:, 0] < 0) | (hand_keypoints[:, 1] < 0)
            invalid_body_mask = (body_keypoints[:, 0] < 0) | (body_keypoints[:, 1] < 0)
            invalid_face_mask = (face_keypoints[:, 0] < 0) | (face_keypoints[:, 1] < 0)

            
            body_keypoints[:,0] = (body_keypoints[:,0] * body_w + body_xmin) / ori_w
            body_keypoints[:,1] = (body_keypoints[:,1] * body_h + body_ymin) / ori_h
            
            hand_keypoints[:,0] = (hand_keypoints[:,0] * body_w + body_xmin) / ori_w
            hand_keypoints[:,1] = (hand_keypoints[:,1] * body_h + body_ymin) / ori_h
            
            face_keypoints[:,0] = (face_keypoints[:,0] * body_w + body_xmin) / ori_w
            face_keypoints[:,1] = (face_keypoints[:,1] * body_h + body_ymin) / ori_h
            
            hand_keypoints[invalid_hand_mask] = -1
            body_keypoints[invalid_body_mask] = -1
            face_keypoints[invalid_face_mask] = -1 

            frame_info_dict['hand'] = hand_keypoints.tolist() # (N, 3) list of (x, y, z). right + left hand
            frame_info_dict['body']  = body_keypoints.tolist()
            frame_info_dict['face']  = face_keypoints.tolist()
            
            clip_info_dict['keyframes'][formated_frame_id] = frame_info_dict

        cap.release()
        return clip_info_dict

    def dump_anno(self, start_idx=0, end_idx=None, debug_anno_dir = None):
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

        # self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=self.keypoints_threshold)
        # self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)
        # self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)


        if end_idx is None:
            end_idx = len(self.annno_paths)
        elif end_idx > len(self.annno_paths):
            end_idx = len(self.annno_paths)
            
        cnt_clip = 0
        
        to_be_processed_ids = list(range(start_idx, end_idx))
        
        if self.debug:
            if debug_anno_dir is not None:
                self.new_anno_dir = debug_anno_dir
            else:
                self.new_anno_dir = 'temp2/anno_debug'
            os.makedirs(self.new_anno_dir, exist_ok=True)
        
        for i, idx in enumerate(to_be_processed_ids):
            json_path = self.annno_paths[idx]
            
            json_path = json_path.replace('/annos/', '/annos_2D/')
            
            json_name = os.path.basename(json_path)
            video_name = json_name.replace('_anno.json', '_frames.mp4')
            
            video_filepath = os.path.join(self.video_dir, video_name)

            print(f"Processing clip {video_name} {i + start_idx}/{end_idx}")

            
            ### load clip info from json file
            with open(json_path, 'r') as f:
                clip_info_dict = json.load(f)

            # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")
            
            # Sample frames with hand filtering
            clip_info_dict = self.process_videos(video_filepath, clip_info_dict, print_flag=True)

            
            new_json_filepath = os.path.join(self.new_anno_dir, json_name)
            # json_filepath = 'youtubeASL_anno.json'
            if os.path.exists(new_json_filepath):
                try:
                    os.remove(new_json_filepath)
                except FileNotFoundError:
                    print(f"Warning: {new_json_filepath} was already deleted by another process.")
            
            with open(new_json_filepath, 'w') as f:
                json.dump(clip_info_dict, f, indent=4, ensure_ascii=False)
                

            cnt_clip += 1
            if self.debug and cnt_clip >= 3:
                break

    def parse_json_visualize(self, frame_dir='youtubeASL_frames', anno_json_dir = 'youtubeASL_anno', output_dir='output'):
        
        """
        Parse the JSON file and visualize the keypoints on the frames.
        Args:
            frame_dir (str): Directory containing the frames.
            anno_json_dir (str): Directory containing the annotation JSON file.
            output_dir (str): Directory to save the output images with keypoints.
        """
        import json
        json_files = os.listdir(anno_json_dir)
        for json_file in json_files:
            # Assuming only one JSON file in the directory
            json_filepath = os.path.join(anno_json_dir, json_file)
            with open(json_filepath, 'r') as f:
                anno_data = json.load(f)

            video_path = json_filepath.replace('_anno.json', '_frames.mp4')
            video_path = video_path.replace(anno_json_dir, frame_dir)
            print(f"Visualizing keypoints for video: {video_path}")
            
            ## read all frames from video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video {video_path}")
                return
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            print(f"Total frames in video: {len(frames)}")
            
            os.makedirs(output_dir, exist_ok=True)

            cnt = 0
            video_name = anno_data['video_name'] 
            for frame_name, frame_info in anno_data['keyframes'].items():
                
                #
                frame_id = int(frame_name)
                new_frame_name = f'{frame_name}.jpg'
                frame = frames[cnt]
                # print('frame_rgb.shape:', frame_rgb.shape)
                # print(f'frame_rgb.dtype:', frame_rgb.dtype)
                hand_kp = frame_info['hand']
                body_kp = frame_info['body']
                face_kp = frame_info['face']
                h, w, _ = frame.shape
                frame = frame.copy()
                # print('hand_kp:', hand_kp)
                for (x, y, z) in hand_kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)
                for (x, y, z) in body_kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x * w), int(y * h)), 1, (255, 0, 0), -1)
                for (x, y, z) in face_kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 0, 255), -1)
                
                cv2.imwrite(f'{output_dir}/{video_name}_{new_frame_name}', frame)
                cnt += 1

                
        
        
                
def main_dump_anno_json(args):
    debug = args.debug
    split = args.split if hasattr(args, 'split') else 'train'
    print(f"Running in debug mode: {debug}, split: {split}")
    

    dataset = MediapipePose(split = split, debug=debug)

    
    if debug:
        output_dir = 'temp/output2'
        os.makedirs(output_dir, exist_ok=True)

    dataset.dump_anno(start_idx=args.start_idx, end_idx=args.end_idx)
    print("Dumped annotation data to JSON file.")

    if debug:
        dataset.parse_json_visualize(frame_dir=dataset.video_dir, anno_json_dir=dataset.new_anno_dir, output_dir=output_dir)
        print(f"Visualized keypoints and saved to {output_dir}")



# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the dataset')
    parser.add_argument('--end_idx', type=int, default=None, help='End index of the dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for limited dataset size')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val/test)')
    args = parser.parse_args()
    # main_sample_examples()
    main_dump_anno_json(args)
    # youTubeASLDumpJson()
    

# Example output: