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
if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)


import functools
print = functools.partial(print, flush=True)


from utils.mediapipe_kpts_mapping import MediapipeKptsMapping




class Mediapipe3DPose:
    def __init__(self, split = 'test', debug = False, num_frames_per_clip = 200, return_frame = False):
        """
        Initialize the YouTube ASL dataset loader for a single video and VTT files.
        
        Args:
            video_dir (str): Directory containing the video ('video.mp4') and transcript (*.vtt) files.
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        
        
        self.video_dir =  '/projects/kosecka/hongrui/dataset/youtubeASL/youtube_ASL/'
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Directory not found: {self.video_dir}")
        self.num_frames_per_clip = num_frames_per_clip
        self.debug = debug


        self.keypoints_threshold = 0.35  # Confidence threshold for keypoint detection
        self.anno_file = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_annotation.txt'
        

    

        print('loading YOLOv8 model...')
        self.yolo_model = YOLO('/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/yolov8n.pt')
        
        # self.yolo_model.fuse()  # Fuse model for faster inference

        print('finished loading YOLOv8 model')
        self.keypoints_threshold = 0.25  # Confidence threshold for keypoint detection
        
        
        self.anno_info_lists = self.parse_annotation(self.anno_file)
        
        if self.debug:
            random.shuffle(self.anno_info_lists)
            self.anno_info_lists = self.anno_info_lists[:3]
        
        print(f"Total clips in dataset: {len(self.anno_info_lists)}")
        

        # # Find video and VTT files
        # self.video_transcript_pairs = self._find_files()
        
        #        
        # Initialize MediaPipe solutions

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
    
    
    def parse_annotation(self, anno_file):
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        
        anno_data = []
        for line in lines:
            video_name, vtt_name, timestamp, text = line.strip().split('||')
            anno_data.append((video_name, vtt_name, timestamp, text))
        
        if self.debug:
            anno_data = anno_data[:200]

        return anno_data


    def _parse_vtt(self, vtt_path: str) -> List[Tuple[float, float, str]]:
        """
        Parse the VTT file to extract time stamps and text.
        
        Args:
            vtt_path (str): Path to the VTT file.
        
        Returns:
            List of tuples (start_time, end_time, text) for valid segments.
        """
        captions = []
        try:
            for caption in webvtt.read(vtt_path):
                start_seconds = self._time_to_seconds(caption.start)
                end_seconds = self._time_to_seconds(caption.end)
                text = caption.text.strip()
                if text:  # Skip empty or invalid text segments
                    captions.append((start_seconds, end_seconds, text))
        except Exception as e:
            print(f"Error parsing VTT file {vtt_path}: {e}")
        return captions

    def _time_to_seconds(self, time_str: str) -> float:
        """
        Convert VTT time string (e.g., "00:00:10.052") to seconds.
        
        Args:
            time_str (str): Time string in format "HH:MM:SS.mmm".
        
        Returns:
            Float representing seconds.
        """
        hours, minutes, seconds = time_str.split(':')
        seconds, milliseconds = seconds.split('.')
        total_seconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000)
        return total_seconds


    

    def _find_files(self) -> List[Tuple[str, str]]:
        """
        Find matching video (.video) and transcript (.vtt) files in the directory.
        
        Returns:
            List of tuples (video_path, vtt_path) with matching prefixes (e.g., 'xx.mp4', 'xx..vtt').
        """
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        video_transcript_pairs = []
        
        for i, video_file in enumerate(video_files):
            base_name = video_file.replace('.mp4', '')
            vtt_files = [f for f in os.listdir(self.video_dir) if f.startswith(base_name) and f.endswith('.vtt')]
            # print(f"Processing video {video_file}, vtt_files: {vtt_files}")
            if vtt_files:
                video_path = os.path.join(self.video_dir, video_file)
                vtt_path = os.path.join(self.video_dir, vtt_files[0])  # Use the first matching VTT file
                video_transcript_pairs.append((video_path, vtt_path))
            
            if i >= 200 and self.debug:
                break
        
        return video_transcript_pairs



    def _detect_keypoints(self, frame_rgb: np.ndarray, det_hand_kpts: bool = True) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """
        Detect body, and face keypoints using MediaPipe.
        
        Args:
            frame_rgb (np.ndarray): RGB frame of shape (height, width, 3).
        
        Returns:
            Tuples of (hand_keypoints, body_keypoints, face_keypoints), each as lists of (x, y, confidence).
        """

        results = self.holistic.process(frame_rgb)

        # --- Hand ---
        left_hand_keypoints = [(-1.0, -1.0)] * len(self.hand_indices)
        right_hand_keypoints = [(-1.0, -1.0)] * len(self.hand_indices)

        if det_hand_kpts and results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                if idx in self.hand_indices:
                    pos = self.hand_indices.index(idx)
                    x, y = landmark.x, landmark.y
                    left_hand_keypoints[pos] = (x, y)

        if det_hand_kpts and results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                if idx in self.hand_indices:
                    pos = self.hand_indices.index(idx)
                    x, y = landmark.x, landmark.y
                    right_hand_keypoints[pos] = (x, y)

        hand_keypoints = right_hand_keypoints + left_hand_keypoints

        # --- Pose (Body) ---
        body_keypoints = [(-1.0, -1.0)] * len(self.body_indices)
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in self.body_indices:
                    pos = self.body_indices.index(idx)
                    x, y = landmark.x, landmark.y
                    body_keypoints[pos] = (x, y)

        # --- Face ---
        face_keypoints = [(-1.0, -1.0)] * len(self.face_indices)
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                if idx in self.face_indices:
                    pos = self.face_indices.index(idx)
                    x, y = landmark.x, landmark.y
                    face_keypoints[pos] = (x, y)

        hand_keypoints = np.array(hand_keypoints, dtype=np.float32)
        body_keypoints = np.array(body_keypoints, dtype=np.float32)
        face_keypoints = np.array(face_keypoints, dtype=np.float32)
        
        return hand_keypoints, body_keypoints, face_keypoints


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



    def fine_sample_frames_with_filter(self, video_path: str, start_time: float, end_time: float, print_flag: bool = False) -> Optional[List[np.ndarray]]:
        """
        Sample frames from a video clip, filter out frames without hand keypoints, and resample.
        
        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
        
        Returns:
            List of sampled frames or None if sampling fails.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if print_flag:
                print(f"Error opening video {video_path}")
            return None, None, None, None, None,  None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # print(f"Video: {video_path}, FPS: {fps}, Total frames: {total_frames}")
        # print(f"Extracted segment: Start={start_time}, End={end_time}")


        # Convert times to frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if start_frame >= total_frames or end_frame > total_frames:
            if print_flag:
                print(f"Time range out of bounds for {video_path}")
            cap.release()
            return None, None,  None, None, None,  None, None, None

        end_frame = min(end_frame, total_frames - 1)
        total_clip_frames = end_frame - start_frame + 1

        if total_clip_frames <= 0:
            cap.release()
            if print_flag:
                print(f"Invalid frame range for {video_path}")
            return None, None, None, None, None,  None, None, None

        # Uniformly sample N + 10 frames initially
        step = 2
        frame_indices = np.arange(start_frame, end_frame, step)
        # print('frame_indices:', frame_indices)
        # frames = []
        selected_frames = []
        frame_index = []

        left_hand_kpts_list = []
        right_hand_kpts_list = []
        face_keypoints_list = []
        body_keypoints_list = []
        

        left_hand_bboxes = []
        right_hand_bboxes = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        det_body_bbox = True

        max_det_frames = len(frame_indices)//3
        
        cnt = 0
        body_bbox = []
        while cnt < len(frame_indices):
            fid = frame_indices[cnt]
        
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                cnt += 1
                continue

            if det_body_bbox:                        
                # Detect body (pose)
                all_kpts = []
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.yolo_model.predict(source=frame_rgb, classes=[0], verbose=False)[0]  # class 0 = person
                person_bboxes = []

                for i in range(len(results.boxes)):
                    if int(results.boxes.cls[i]) == 0:
                        x1, y1, x2, y2 = results.boxes.xyxy[i].tolist()
                        person_bboxes.append([x1, y1, x2, y2])
                if len(person_bboxes) == 0:
                    if print_flag:
                        print(f"{video_path}: No person detected in frame {fid}.")
                    cnt += 10
                    continue
                # 若多人，跳过 clip
                elif len(person_bboxes) > 1:
                    if print_flag:
                        print(f"{video_path}: detected {len(person_bboxes)} persons, selecting largest.")
                    # 选择最大面积的 bbox
                    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in person_bboxes]
                    max_idx = areas.index(max(areas))
                    person_bboxes = [person_bboxes[max_idx]]
                
                # 单人，使用其 bbox
                body_xmin, body_ymin, body_xmax, body_ymax = [int(v) for v in person_bboxes[0]]

                h, w, _ = frame.shape
                body_xmin = max(0, body_xmin)
                body_ymin = max(0, body_ymin)
                body_xmax = min(w, body_xmax)
                body_ymax = min(h, body_ymax)
                body_w = body_xmax - body_xmin
                body_h = body_ymax - body_ymin

                if body_w * body_h < 0.01 * w * h:
                    cnt += 10
                    if print_flag:
                        print(f"Skip {video_path}: person bbox too small.")
                        
                    if cnt > max_det_frames:
                        cap.release()
                        return None, None, None, None, None,  None, None, None
                    continue

                body_bbox = [body_xmin, body_ymin, body_xmax, body_ymax]
                det_body_bbox = False
                
            ori_h, ori_w, _ = frame.shape
            left_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices)
            right_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices)
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

            right_hand_keypoints = hand_keypoints[:21]
            left_hand_keypoints = hand_keypoints[21:]

            



            left_hand_kpts = np.array(left_hand_keypoints)
            right_hand_kpts = np.array(right_hand_keypoints)
            ## find xmin, ymin from these available points with x 0 and y > 0
            valid_left_kpts = left_hand_kpts[left_hand_kpts[:, 0] > 0]
            valid_right_kpts = right_hand_kpts[right_hand_kpts[:, 0] > 0]

            if valid_left_kpts.shape[0] > 0:  # 确保至少有一个合法点
                left_xmin = np.min(valid_left_kpts[:, 0]) * ori_w
                left_ymin = np.min(valid_left_kpts[:, 1]) * ori_h
                left_xmax = np.max(valid_left_kpts[:, 0]) * ori_w
                left_ymax = np.max(valid_left_kpts[:, 1]) * ori_h
            else:
                left_xmin = 0  # 没有合法点，设置为 0
                left_ymin = 0
                left_xmax = 0
                left_ymax = 0

            if valid_right_kpts.shape[0] > 0:  # 确保至少有一个合法点
                right_xmin = np.min(valid_right_kpts[:, 0]) * ori_w
                right_ymin = np.min(valid_right_kpts[:, 1]) * ori_h
                right_xmax = np.max(valid_right_kpts[:, 0]) * ori_w
                right_ymax = np.max(valid_right_kpts[:, 1]) * ori_h
            else:
                right_xmin = 0
                right_ymin = 0
                right_xmax = 0
                right_ymax = 0
            
            cur_left_hand_bbox = [left_xmin, left_ymin, left_xmax, left_ymax]
            cur_right_hand_bbox = [right_xmin, right_ymin, right_xmax, right_ymax]
            previous_left_hand_bbox = left_hand_bboxes[-1] if left_hand_bboxes else [0, 0, 0, 0]
            previous_right_hand_bbox = right_hand_bboxes[-1] if right_hand_bboxes else [0, 0, 0, 0]
            
            ### compute IoU with previous hand bboxes
            def compute_iou(bbox1, bbox2):
                x1 = max(bbox1[0], bbox2[0])
                y1 = max(bbox1[1], bbox2[1])
                x2 = min(bbox1[2], bbox2[2])
                y2 = min(bbox1[3], bbox2[3])

                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

                union_area = bbox1_area + bbox2_area - inter_area
                return inter_area / union_area if union_area > 0 else 0
            
            
            cnt += 1

            if len(frame_indices)/6  < cnt or cnt > 5*len(frame_indices)/6:
                left_hand_iou = compute_iou(cur_left_hand_bbox, previous_left_hand_bbox)
                right_hand_iou = compute_iou(cur_right_hand_bbox, previous_right_hand_bbox)
                if left_hand_iou > 0.9 and right_hand_iou > 0.9:
                    continue

            selected_frames.append(frame_rgb)
            frame_index.append(fid)                    
            left_hand_kpts_list.append(left_hand_keypoints.tolist())
            right_hand_kpts_list.append(right_hand_keypoints.tolist())
            left_hand_bboxes.append(cur_left_hand_bbox)
            right_hand_bboxes.append(cur_right_hand_bbox)
            face_keypoints_list.append(face_keypoints.tolist())
            body_keypoints_list.append(body_keypoints.tolist())

        cap.release()
        return selected_frames, frame_index, left_hand_kpts_list, right_hand_kpts_list, face_keypoints_list, body_keypoints_list, total_clip_frames, body_bbox

    def dump_anno(self, save_img_dir, save_json_dir, start_idx=0, end_idx=None):
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
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_json_dir, exist_ok=True)

        if end_idx is None:
            end_idx = len(self.anno_info_lists)
        elif end_idx > len(self.anno_info_lists):
            end_idx = len(self.anno_info_lists)  # Ensure end_idx does not exceed the list length
            
        cnt_clip = 0

        to_be_processed_ids = np.arange(start_idx, end_idx)

        for i, idx in enumerate(to_be_processed_ids):
            clip_info_dict = {}
            clip_id = f'{idx:08d}'

            '''
            wGkZ5xID3_8.mp4||wGkZ5xID3_8.en.vtt||00:04:39.880 --> 00:04:42.780||If you want to book a hotel room,
            wGkZ5xID3_8.mp4||wGkZ5xID3_8.en.vtt||00:04:43.000 --> 00:04:47.580||reservations happen 1 year and 1 day in advance.
            '''
            video_name, vtt_name, timestamp, text = self.anno_info_lists[idx]
            video_name_prefix = os.path.splitext(video_name)[0]
            
            
            # video_name_prefix, scentence_id, start_time, end_time, text = self.anno_info_lists[idx]
            
            
            scentence_id = f'{idx:08d}'
            squeezed_frame_path = os.path.join(save_img_dir,  f'{video_name_prefix}_SID{scentence_id}_frames.mp4')
            if os.path.exists(squeezed_frame_path):
                print(f"Warning: {squeezed_frame_path} already exists, skipping.")
                continue
            
            print(f"Processing clip {video_name} {i + start_idx}/{end_idx}")
            
            
            # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")
            start_time, end_time = timestamp.split(' --> ')
            start_time = self._time_to_seconds(start_time)
            end_time = self._time_to_seconds(end_time)

            video_path = os.path.join(self.video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Warning: video {video_path} does not exist.")
                continue
            
                
            
            # Sample frames with hand filtering
            selected_frames, frame_index, left_hand_kpts_list, right_hand_kpts_list, \
                face_keypoints_list, body_keypoints_list, total_clip_frames, body_bbox = self.fine_sample_frames_with_filter(video_path, start_time, 
                                                                                           end_time, print_flag=True)


            if selected_frames is None:
                print(f"Warning: No valid frames found for {video_path} in {start_time} ~ {end_time}.")
                continue
            if len(selected_frames) <= 5:
                print(f"Warning: Not enough valid frames for {video_path} in {start_time} ~ {end_time}. Found {len(selected_frames)} frames.")
                continue
            
            height, width, _ = selected_frames[0].shape

            clip_info_dict['video_name'] = video_name
            clip_info_dict['clip_id'] = clip_id
            clip_info_dict['text'] = text
            clip_info_dict['start_time'] = start_time
            clip_info_dict['end_time'] = end_time
            clip_info_dict['height'] = height
            clip_info_dict['width'] = width
            clip_info_dict['body_bbox'] = body_bbox
            
            clip_info_dict['filtered_frame_num'] = len(selected_frames)
            clip_info_dict['keyframes'] = {}
            
            
            # Process each frame for keypoints

            for i, frame_id in enumerate(frame_index):
                frame_info_dict = {}

                formated_frame_id = f'{frame_id:06d}'
                # new_frame_id = f'{i:06d}'
                
                # frame_name = new_frame_id
                
                face_kp = face_keypoints_list[i]
                body_kp = body_keypoints_list[i]

                # print('len(face_kp):', len(face_kp))
                # print('hand_kp:', hand_kp)
                hand_kp = right_hand_kpts_list[i] + left_hand_kpts_list[i]
                # cv2.imwrite(f'{save_img_dir}/{frame_name}', frame_rgb[:,:,::-1])
                
                # print(f"    Saved frame {self.video_dir}/{frame_name}")
                # cv2.imwrite(f'output/{frame_name}', frame_rgb[:,:,::-1])
                frame_info_dict['hand'] = hand_kp
                frame_info_dict['body'] = body_kp
                frame_info_dict['face'] = face_kp
                frame_info_dict['original_frame_id'] = formated_frame_id
                # frame_info_dict['frame_name'] = frame_name
                clip_info_dict['keyframes'][formated_frame_id] = frame_info_dict
                # print()
            
            
            # anno_json[f"clip_{clip_id}"] = clip_info_dict
            # break
            # if idx > 2:
            #     break
            
            import json
            json_filepath = os.path.join(save_json_dir, f'{video_name_prefix}_SID{scentence_id}_anno.json')
            # json_filepath = 'youtubeASL_anno.json'
            if os.path.exists(json_filepath):
                try:
                    os.remove(json_filepath)
                except FileNotFoundError:
                    print(f"Warning: {json_filepath} was already deleted by another process.")
            
            with open(json_filepath, 'w') as f:
                json.dump(clip_info_dict, f, indent=4, ensure_ascii=False)
                
            height, width, _ = selected_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 3
            
            
            writer = cv2.VideoWriter(squeezed_frame_path, fourcc, fps, (width, height))

            for frame in selected_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                writer.write(frame)  # 确保 frame 是 BGR 格式
            writer.release()
            # print(f"✅ 成功保存视频到: {output_path}")


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
                for (x, y, _) in hand_kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)
                for (x, y, _) in body_kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x * w), int(y * h)), 1, (255, 0, 0), -1)
                for (x, y, _) in face_kp:
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 0, 255), -1)
                
                cv2.imwrite(f'{output_dir}/{video_name}_{new_frame_name}', frame)
                cnt += 1

                
        
        
                
def main_dump_anno_json(args):
    debug = args.debug
    split = args.split if hasattr(args, 'split') else 'train'
    print(f"Running in debug mode: {debug}, split: {split}")
    
    

    dataset = Mediapipe3DPose(split = split, debug=debug)

    
    if debug:
        save_img_dir = f'temp2/{split}/frames'
        save_anno_dir = f'temp2/{split}/annos'
        output_dir = 'temp2/output2'
        os.makedirs(output_dir, exist_ok=True)
    else:
        # save_img_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_frames'
        # save_anno_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno'
        # save_img_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_frames'
        # save_anno_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_anno'
        save_img_dir = f'/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/frames'
        save_anno_dir = f'/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/annos'

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_anno_dir, exist_ok=True)
    

    dataset.dump_anno(save_img_dir, save_anno_dir, start_idx=args.start_idx, end_idx=args.end_idx)
    print("Dumped annotation data to JSON file.")

    if debug:
        dataset.parse_json_visualize(frame_dir=save_img_dir, anno_json_dir=save_anno_dir, output_dir=output_dir)
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