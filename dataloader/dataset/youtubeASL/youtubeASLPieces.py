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

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)
    

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping

class YouTubeASLPieces(Dataset):
    def __init__(self, video_dir: str = None, num_frames_per_clip: int = 30, target_img_size = (224, 224), debug = False):
        """
        Initialize the YouTube ASL dataset loader for a single video and VTT files.
        
        Args:
            video_dir (str): Directory containing the video ('video.mp4') and transcript (*.vtt) files.
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.video_dir = video_dir if video_dir is not None else '/scratch/rhong5/dataset/youtube_ASL/'
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Directory not found: {self.video_dir}")
        self.num_frames_per_clip = num_frames_per_clip
        self.target_img_size = target_img_size
        self.debug = debug
        self.keypoints_threshold = 0.3
        self.anno_file = '/scratch/rhong5/dataset/youtubeASL_annotation.txt'

        self.anno_data = self.parse_annotation(self.anno_file)
        
        print(f"Total clips in dataset: {len(self.anno_data)}")

        # # Find video and VTT files
        # self.video_transcript_pairs = self._find_files()
        
        #        
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=self.keypoints_threshold)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=self.keypoints_threshold)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=self.keypoints_threshold)
        
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

    def uniform_sample_frames_with_hand_filter(self, video_path: str, start_time: float, end_time: float, print_flag: bool = False) -> Optional[List[np.ndarray]]:
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
            return None, None

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
            return None, None

        end_frame = min(end_frame, total_frames - 1)
        total_clip_frames = end_frame - start_frame + 1

        if total_clip_frames <= 0:
            cap.release()
            if print_flag:
                print(f"Invalid frame range for {video_path}")
            return None, None

        # Uniformly sample N + 10 frames initially
        initial_num_frames = self.num_frames_per_clip + 10
        frame_indices = np.linspace(start_frame, end_frame, initial_num_frames, dtype=int)
        # frames = []
        selected_frames = []
        frame_index = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect hands using MediaPipe
                results = self.mp_hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    # print(f"Detected hands in frame {idx}, landmarks: {len(results.multi_hand_landmarks)}")
                    # print(f"Hand landmarks: {results.multi_hand_landmarks}")
                    ## pad frame_rgb to a square image based on the larger dimension
                    h, w, _ = frame_rgb.shape
                    if h > w:
                        pad = (h-w)//2
                        frame_rgb = cv2.copyMakeBorder(frame_rgb, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    elif w > h:
                        pad = (w-h)//2
                        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    selected_frames.append(frame_rgb)
                    frame_index.append(idx)
                # frames.append(frame_rgb)
            else:
                break
        cap.release()

        if not selected_frames:
            if print_flag:
                print(f"No hand-detected frames in {video_path}") 
            return None, None
        if print_flag:
            print(f"Detected {len(selected_frames)} / {total_clip_frames} frames, in {video_path}")
        # Uniformly sample num_frames from hand-detected frames
        if len(selected_frames) <= self.num_frames_per_clip:
            sampled_frames = selected_frames
            sampled_frames_index = frame_index
        else:
            sample_indices = np.linspace(0, len(selected_frames) - 1, self.num_frames_per_clip, dtype=int)
            sampled_frames = [selected_frames[i] for i in sample_indices]
            sampled_frames_index = [frame_index[i] for i in sample_indices]

        return sampled_frames, sampled_frames_index

    def _detect_keypoints(self, frame_rgb: np.ndarray, frame_id: str = None, det_hand_kpts: bool = True) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """
        Detect hand, body, and face keypoints using MediaPipe.
        
        Args:
            frame_rgb (np.ndarray): RGB frame of shape (height, width, 3).
        
        Returns:
            Tuples of (hand_keypoints, body_keypoints, face_keypoints), each as lists of (x, y, confidence).
        """

        if det_hand_kpts: # if hand keypoints have not been detected,
            # Detect hands
            left_hand_keypoints, right_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices), [(-1.0, -1.0)]*len(self.hand_indices)

            results_hands = self.mp_hands.process(frame_rgb)

            if not frame_id is None:
                # print(f'Processing frame {frame_id}')
                if results_hands.multi_hand_landmarks is None:
                    print('results_hands.multi_hand_landmarks:', results_hands.multi_hand_landmarks)
                # print('results_hands.multi_handedness:', results_hands.multi_handedness)

            if results_hands.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)):
                    hand_label = handedness.classification[0].label  # "Left" or "Right"

                    hand_keypoints = [(-1.0, -1.0)] * len(self.hand_indices)
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        if idx in self.hand_indices:  # 只选取 21 个关键点
                            # print('idx:', idx, 'self.hand_indices:', self.hand_indices)
                            pos = self.hand_indices.index(idx)  # Map to body indices (0–8)
                            # x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                            x, y = landmark.x, landmark.y
                            # confidence = 1.0  # MediaPipe Hands 没有 visibility
                            hand_keypoints[pos] = (x, y)
                            

                    
                    if hand_label == "Left":
                        left_hand_keypoints = hand_keypoints  # 确保左手存储
                    else:
                        right_hand_keypoints = hand_keypoints  # 确保右手存储

            hand_keypoints = right_hand_keypoints + left_hand_keypoints
        else:
            hand_keypoints = None

        # Detect body (pose)
        body_keypoints = [(-1.0, -1.0)] * len(self.body_indices)
        results_pose = self.mp_pose.process(frame_rgb)
        if results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if idx in self.body_indices:  # Use only specified body indices (0–8)
                    pos = self.body_indices.index(idx)
                    # x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                    x, y = landmark.x, landmark.y
                    # confidence = landmark.visibility
                    body_keypoints[pos] = (x, y)

        # Detect face
        face_keypoints = [(-1.0, -1.0)] * len(self.face_indices)  # 预填充 18 个点
        results_face = self.mp_face_mesh.process(frame_rgb)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in self.face_indices:  # Use only specified face indices (0–17)
                        pos = self.face_indices.index(idx)
                        # x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                        # confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                        x, y = landmark.x, landmark.y
                        face_keypoints[pos] = (x, y)

        return hand_keypoints, body_keypoints, face_keypoints

    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.anno_data)

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

        # self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=self.keypoints_threshold)
        # self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)
        # self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)

        video_name, vtt_name, timestamp, text = self.anno_data[idx]
        video_name_prefix = video_name.split('.')[0]

        start_time, end_time = timestamp.split(' --> ')
        # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")

        start_time = self._time_to_seconds(start_time)
        end_time = self._time_to_seconds(end_time)

        video_path = os.path.join(self.video_dir, video_name)
        
        
        # Sample frames with hand filtering
        sampled_frames_rgb, sampled_frames_index = self.uniform_sample_frames_with_hand_filter(video_path, start_time, end_time)
        # print(f"Sampled frames: {len(sampled_frames)}, target frames: {self.num_frames_per_clip}")
    

        if sampled_frames_rgb is None or len(sampled_frames_rgb) == 0:
            return self.__getitem__(np.random.randint(0, len(self.anno_data)))

        # Process each frame for keypoints
        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        for frame_rgb in sampled_frames_rgb:
            hand_kp, body_kp, face_kp = self._detect_keypoints(frame_rgb)
            # print('len(face_kp):', len(face_kp))
            hand_keypoints_all.append(hand_kp)
            body_keypoints_all.append(body_kp)
            face_keypoints_all.append(face_kp)

        # print('face_keypoints_all:', face_keypoints_all)
        # print('face_keypoints_all:', face_keypoints_all)
        hand_keypoints_all = np.array(hand_keypoints_all) # (N, 42, 2)
        body_keypoints_all = np.array(body_keypoints_all) # (N, 9, 2)
        face_keypoints_all = np.array(face_keypoints_all) # (N, 18, 2)
        sampled_frames_rgb = np.array(sampled_frames_rgb) # (N, h, w, 3)

        pad_len = self.num_frames_per_clip - sampled_frames_rgb.shape[0]  # 计算填充长度

        if pad_len > 0:  # 仅当需要填充时执行
            sampled_frames_rgb = np.concatenate([sampled_frames_rgb, np.zeros((pad_len,) + sampled_frames_rgb.shape[1:])], axis=0)
            hand_keypoints_all = np.concatenate([hand_keypoints_all, np.full((pad_len, 2 * len(self.hand_indices), 2), -1)], axis=0)
            body_keypoints_all = np.concatenate([body_keypoints_all, np.full((pad_len, len(self.body_indices), 2), -1)], axis=0)
            face_keypoints_all = np.concatenate([face_keypoints_all, np.full((pad_len, len(self.face_indices), 2), -1)], axis=0)



        # Stack frames into a tensor
        frames_tensor = torch.from_numpy(sampled_frames_rgb).float() / 255.0  # 归一化到 [0, 1]
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # 变为 (N, C, H, W)

        frames_resized = F.interpolate(frames_tensor, size=self.target_img_size, mode='bilinear', align_corners=False) # (N, C, 224, 224)


        # Create keypoints dictionary
        keypoints_dict = {
            'hand': hand_keypoints_all,
            'body': body_keypoints_all,
            'face': face_keypoints_all
        }

        # print(f"Frames tensor shape: {frames_tensor.shape}, Text: {text}")


        return (frames_resized, text, keypoints_dict)
    
    def iou_filter(self, selected_frames, left_hand_bboxes, right_hand_bboxes, threshold=0.4):

        def compute_iou(bbox1, bbox2):
            """
            计算两个 bounding box 之间的 IoU (Intersection over Union)
            bbox 格式: [xmin, ymin, xmax, ymax]
            """
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)  # 交集面积
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # BBox1 面积
            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # BBox2 面积

            union_area = bbox1_area + bbox2_area - inter_area  # 并集面积
            return inter_area / union_area if union_area > 0 else 0  # 避免除 0 错误
        
        filtered_frames = [selected_frames[0]]  # 存储最终的关键帧
        filtered_left_bboxes = [left_hand_bboxes[0]]
        filtered_right_bboxes = [right_hand_bboxes[0]]

        indices = [0]  # 存储最终的关键帧索引
        cur_idx = 0  # 当前帧索引

        for i in range(1, len(selected_frames)-2):  # 遍历后续帧
            left_iou = compute_iou(left_hand_bboxes[cur_idx], left_hand_bboxes[i])
            right_iou = compute_iou(right_hand_bboxes[cur_idx], right_hand_bboxes[i])

            if left_iou <= threshold or right_iou <= threshold:  
                # 只要有一个手的 IoU 小于等于 0.7，就保留该帧
                filtered_frames.append(selected_frames[i])
                
                filtered_left_bboxes.append(left_hand_bboxes[i])
                filtered_right_bboxes.append(right_hand_bboxes[i])
                cur_idx = i  # 更新当前帧索引
                indices.append(i)
        return filtered_frames, indices

    def fine_sample_frames_with_filter(self, video_path: str, start_time: float, end_time: float, drop_last: int, print_flag: bool = False) -> Optional[List[np.ndarray]]:
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
            return None, None, None, None, None

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
            return None, None,  None, None, None

        end_frame = min(end_frame, total_frames - 1)
        total_clip_frames = end_frame - start_frame + 1

        if total_clip_frames <= 0:
            cap.release()
            if print_flag:
                print(f"Invalid frame range for {video_path}")
            return None, None, None, None, None

        # Uniformly sample N + 10 frames initially
        initial_num_frames = self.num_frames_per_clip + 15 if self.num_frames_per_clip + 15 < total_clip_frames else total_clip_frames
        frame_indices = np.linspace(start_frame, end_frame - drop_last, initial_num_frames, dtype=int)
        # print('frame_indices:', frame_indices)
        # frames = []
        selected_frames = []
        frame_index = []
        left_hand_bboxes = []
        right_hand_bboxes = []
        left_hand_kpts_list = []
        right_hand_kpts_list = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        det_body_bbox = True
        for m, fid in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                continue

            if det_body_bbox:                        
                # Detect body (pose)
                all_kpts = []
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_pose = self.mp_pose.process(frame_rgb)
                h, w, _ = frame.shape
                # print(f'h x w: {h} x {w}')
                if results_pose.pose_landmarks:
                    for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                        all_kpts.append((int(w*landmark.x), int(h*landmark.y)))
                else:
                    continue
            
                all_kpts = np.array(all_kpts)
                body_xmin, body_ymin = np.min(all_kpts, axis=0)
                body_xmax, body_ymax = np.max(all_kpts, axis=0)
                # cv2.imwrite('temp.jpg', frame)
                # print(f'1 body_xmin: {body_xmin}, body_ymin: {body_ymin}, body_xmax: {body_xmax}, body_ymax: {body_ymax}')

                body_w = body_xmax - body_xmin
                body_h = body_ymax - body_ymin
                body_xmin = max(0, body_xmin - body_w//2)
                body_ymin = max(0, body_ymin - body_h//3)
                body_xmax = min(w, body_xmax + body_w//2)
                body_ymax = min(h, body_ymax + body_h//3)
                body_bbox = [body_xmin, body_ymin, body_xmax, body_ymax]
                # print(f'2 body_xmin: {body_xmin}, body_ymin: {body_ymin}, body_xmax: {body_xmax}, body_ymax: {body_ymax}')
                
                body_w = body_xmax - body_xmin
                body_h = body_ymax - body_ymin

                if body_w * body_h < 0.01 * w * h:
                    continue
                det_body_bbox = False
                
            ori_h, ori_w, _ = frame.shape
            left_hand_keypoints, right_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices), [(-1.0, -1.0)]*len(self.hand_indices)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print('body_bbox:', body_bbox)
            
            body_rgb = frame_rgb[body_ymin:body_ymax, body_xmin:body_xmax].copy()

            # Detect hands using MediaPipe
            results_hands = self.mp_hands.process(body_rgb)
            if results_hands.multi_hand_landmarks:
                # print(f"Detected hands in frame {idx}, landmarks: {len(results.multi_hand_landmarks)}")
                # print(f"Hand landmarks: {results.multi_hand_landmarks}")
                ## pad frame_rgb to a square image based on the larger dimension
                h, w, _ = body_rgb.shape
                # if h > w:
                #     pad = (h-w)//2
                #     frame_rgb = cv2.copyMakeBorder(frame_rgb, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # elif w > h:
                #     pad = (w-h)//2
                #     frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                
                h, w = body_rgb.shape[:2]
                

                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)):
                    hand_label = handedness.classification[0].label  # "Left" or "Right"

                    hand_keypoints = [(-1.0, -1.0)] * len(self.hand_indices)
                    for ix, landmark in enumerate(hand_landmarks.landmark):
                        if ix in self.hand_indices:  # 只选取 21 个关键点
                            # print('idx:', idx, 'self.hand_indices:', self.hand_indices)
                            pos = self.hand_indices.index(ix)  # Map to body indices (0–8)
                            # x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                            # x, y = landmark.x, landmark.y
                            x = (landmark.x * w  + body_xmin ) / ori_w
                            y = (landmark.y * h + body_ymin) / ori_h
                            
                            # confidence = 1.0  # MediaPipe Hands 没有 visibility
                            hand_keypoints[pos] = (x, y)
                            
                    
                    if hand_label == "Left":
                        left_hand_keypoints = hand_keypoints  # 确保左手存储
                    else:
                        right_hand_keypoints = hand_keypoints  # 确保右手存储
                

                left_hand_kpts = np.array(left_hand_keypoints)
                right_hand_kpts = np.array(right_hand_keypoints)
                ## find xmin, ymin from these available points with x 0 and y > 0
                valid_left_kpts = left_hand_kpts[left_hand_kpts[:, 0] > 0]
                valid_right_kpts = right_hand_kpts[right_hand_kpts[:, 0] > 0]
                if valid_left_kpts.shape[0] > 0:  # 确保至少有一个合法点
                    left_xmin = np.min(valid_left_kpts[:, 0]) * ori_w
                    left_ymin = np.min(valid_left_kpts[:, 1]) * ori_h
                    left_xmax = np.max(left_hand_kpts[:, 0]) * ori_w
                    left_ymax = np.max(left_hand_kpts[:, 1]) * ori_h
                else:
                    left_xmin = 0  # 没有合法点，设置为 0
                    left_ymin = 0
                    left_xmax = 0
                    left_ymax = 0

                if valid_right_kpts.shape[0] > 0:  # 确保至少有一个合法点
                    right_xmin = np.min(valid_right_kpts[:, 0]) * ori_w
                    right_ymin = np.min(valid_right_kpts[:, 1]) * ori_h
                    right_xmax = np.max(right_hand_kpts[:, 0]) * ori_w
                    right_ymax = np.max(right_hand_kpts[:, 1]) * ori_h
                else:
                    right_xmin = 0
                    right_ymin = 0
                    right_xmax = 0
                    right_ymax = 0
                
                selected_frames.append(frame_rgb)
                frame_index.append(fid)                    
                left_hand_kpts_list.append(left_hand_keypoints)
                right_hand_kpts_list.append(right_hand_keypoints)
                left_hand_bboxes.append([left_xmin, left_ymin, left_xmax, left_ymax])
                right_hand_bboxes.append([right_xmin, right_ymin, right_xmax, right_ymax])
                
        cap.release()

        # print(f'frame_index:', frame_index)
        if not selected_frames:
            if print_flag:
                print(f"No hand-detected frames in {video_path}") 
            return None, None, None, None, None
        if print_flag:
            print(f"Detected {len(selected_frames)} / {total_clip_frames} frames, in {video_path}")
        # Uniformly sample num_frames from hand-detected frames
        if len(selected_frames) <= self.num_frames_per_clip:
            filtered_frames = selected_frames
            filtered_frames_index = frame_index
            filtered_left_hand_kpts_list = left_hand_kpts_list
            filtered_right_hand_kpts_list = right_hand_kpts_list
        else:
            filtered_frames, indices = self.iou_filter(selected_frames, left_hand_bboxes, right_hand_bboxes)
            filtered_frames_index = [frame_index[i] for i in indices]
            filtered_left_hand_kpts_list = [left_hand_kpts_list[i] for i in indices]
            filtered_right_hand_kpts_list = [right_hand_kpts_list[i] for i in indices]

            if len(filtered_frames) > self.num_frames_per_clip:
                ## uniform sample
                sample_indices = np.linspace(0, len(filtered_frames) - 2, self.num_frames_per_clip, dtype=int) ## discard the last 2 frames
                filtered_frames = [filtered_frames[i] for i in sample_indices]
                filtered_frames_index = [filtered_frames_index[i] for i in sample_indices]
                filtered_left_hand_kpts_list = [filtered_left_hand_kpts_list[i] for i in sample_indices]
                filtered_right_hand_kpts_list = [filtered_right_hand_kpts_list[i] for i in sample_indices]


        return filtered_frames, filtered_frames_index, filtered_left_hand_kpts_list, filtered_right_hand_kpts_list, body_bbox
    
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
        anno_json = {}
        end_idx = len(self.anno_data) if end_idx is None else end_idx
        for idx in range(start_idx, end_idx):
            clip_info_dict = {}
            clip_id = f'{idx:08d}'

            video_name, vtt_name, timestamp, text = self.anno_data[idx]
            video_name_prefix = video_name.split('.')[0]

            print(f"Processing clip {video_name} {idx}/{end_idx}")

            start_time, end_time = timestamp.split(' --> ')
            # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")

            start_time = self._time_to_seconds(start_time)
            end_time = self._time_to_seconds(end_time)

            video_path = os.path.join(self.video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Warning: video {video_name} {video_path} does not exist.")
                continue
            
                
            if text.endswith('.'):
                drop_last = 12
            else:
                drop_last = 3

            
            # Sample frames with hand filtering
            filtered_frames, filtered_frames_index, filtered_left_hand_kpts_list, filtered_right_hand_kpts_list, body_bbox = self.fine_sample_frames_with_filter(video_path, start_time, end_time, drop_last, print_flag=False)
            # print(f"Sampled frames: {len(filtered_frames)}, {filtered_frames_index}")
        

            if filtered_frames is None or len(filtered_frames) == 0:
                continue

            clip_info_dict['video_name'] = video_name
            clip_info_dict['clip_id'] = clip_id
            clip_info_dict['text'] = text
            clip_info_dict['start_time'] = start_time
            clip_info_dict['end_time'] = end_time
            clip_info_dict['keyframes'] = {}
            
            body_xmin, body_ymin, body_xmax, body_ymax = body_bbox
            # Process each frame for keypoints
            for i, frame_rgb in enumerate(filtered_frames):
                frame_info_dict = {}
                frame_id = filtered_frames_index[i]
                formated_frame_id = f'{frame_id:08d}'
                frame_name = f'{video_name_prefix}_C{clip_id}_fid_{formated_frame_id}.jpg'
                
                ori_h, ori_w, _ = frame_rgb.shape
                body_rgb = frame_rgb[body_ymin:body_ymax, body_xmin:body_xmax].copy()
                _, body_kp, face_kp = self._detect_keypoints(body_rgb, frame_id, det_hand_kpts=False)
                body_kp = np.array(body_kp)
                face_kp = np.array(face_kp)

                invalid_body_kp = body_kp < 0
                invalid_face_kp = face_kp < 0

                body_kp[:, 0] = (body_kp[:, 0] * (body_xmax - body_xmin) + body_xmin) / ori_w
                body_kp[:, 1] = (body_kp[:, 1] * (body_ymax - body_ymin) + body_ymin) / ori_h
                
                face_kp[:, 0] = (face_kp[:, 0] * (body_xmax - body_xmin) + body_xmin) / ori_w
                face_kp[:, 1] = (face_kp[:, 1] * (body_ymax - body_ymin) + body_ymin) / ori_h

                face_kp[invalid_face_kp] = -1
                body_kp[invalid_body_kp] = -1

                face_kp = face_kp.tolist()
                body_kp = body_kp.tolist()

                # print('len(face_kp):', len(face_kp))
                # print('hand_kp:', hand_kp)
                hand_kp = filtered_right_hand_kpts_list[i] + filtered_left_hand_kpts_list[i]
                cv2.imwrite(f'{save_img_dir}/{frame_name}', frame_rgb[:,:,::-1])
                # print(f"    Saved frame {self.video_dir}/{frame_name}")
                # cv2.imwrite(f'output/{frame_name}', frame_rgb[:,:,::-1])
                frame_info_dict['hand'] = hand_kp
                frame_info_dict['body'] = body_kp
                frame_info_dict['face'] = face_kp
                # frame_info_dict['frame_name'] = frame_name
                clip_info_dict['keyframes'][frame_name] = frame_info_dict
                # print()
            
            
            # anno_json[f"clip_{clip_id}"] = clip_info_dict
            # break
            # if idx > 2:
            #     break
            
            import json
            json_filepath = os.path.join(save_json_dir, f'youtubeASL_Clip{clip_id}_anno.json')
            # json_filepath = 'youtubeASL_anno.json'
            if os.path.exists(json_filepath):
                try:
                    os.remove(json_filepath)
                except FileNotFoundError:
                    print(f"Warning: {json_filepath} was already deleted by another process.")
            
            with open(json_filepath, 'w') as f:
                json.dump(clip_info_dict, f, indent=4, ensure_ascii=False)

    def uniform_sample(self, video_path: str, start_time: float, end_time: float, print_flag: bool = False) -> Optional[List[np.ndarray]]:
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
            return None, None

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
            return None, None

        end_frame = min(end_frame, total_frames - 1)
        total_clip_frames = end_frame - start_frame + 1

        if total_clip_frames <= 0:
            cap.release()
            if print_flag:
                print(f"Invalid frame range for {video_path}")
            return None, None

        # Uniformly sample N + 10 frames initially
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames_per_clip, dtype=int)

        selected_frames = []
        frame_index = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for m, fid in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {fid}")
                continue

            selected_frames.append(frame)
            frame_index.append(fid)
        cap.release()
        return selected_frames, frame_index



    def sample_draw_kpts(self, save_img_dir, start_idx = 0, end_idx = None):
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
        # os.makedirs(save_json_dir, exist_ok=True)
        # anno_json = {}
        end_idx = len(self.anno_data) if end_idx is None else end_idx
        available_cnt = 0
        for idx in range(start_idx, end_idx):
            clip_info_dict = {}
            clip_id = f'{idx:08d}'

            video_name, vtt_name, timestamp, text = self.anno_data[idx]
            video_name_prefix = video_name.split('.')[0]

            print(f"Processing clip {video_name} {idx}/{end_idx}")

            start_time, end_time = timestamp.split(' --> ')
            # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")

            start_time = self._time_to_seconds(start_time)
            end_time = self._time_to_seconds(end_time)

            video_path = os.path.join(self.video_dir, video_name)
            if not os.path.exists(video_path):
                continue
            
                
            if text.endswith('.'):
                drop_last = 12
            else:
                drop_last = 3

            
            # Sample frames with hand filtering
            filtered_frames, filtered_frames_index, filtered_left_hand_kpts_list, filtered_right_hand_kpts_list, body_bbox = self.fine_sample_frames_with_filter(video_path, start_time, end_time, drop_last, print_flag=False)
            # print(f"Sampled frames: {len(filtered_frames)}, {filtered_frames_index}")
        

            if not filtered_frames is None and len(filtered_frames)> 0:
                print(f"Filtered Detected {len(filtered_frames)} / {len(filtered_frames)} frames, in {video_path}")
                body_xmin, body_ymin, body_xmax, body_ymax = body_bbox
                # Process each frame for keypoints
                for i, frame_rgb in enumerate(filtered_frames):
                    frame_info_dict = {}
                    frame_id = filtered_frames_index[i]
                    formated_frame_id = f'{frame_id:08d}'
                    frame_name = f'{video_name_prefix}_C{clip_id}_fid_{formated_frame_id}.jpg'
                    
                    ori_h, ori_w, _ = frame_rgb.shape
                    body_rgb = frame_rgb[body_ymin:body_ymax, body_xmin:body_xmax].copy()
                    _, body_kp, face_kp = self._detect_keypoints(body_rgb, frame_id, det_hand_kpts=False)
                    body_kp = np.array(body_kp)
                    face_kp = np.array(face_kp)

                    invalid_body_kp = body_kp < 0
                    invalid_face_kp = face_kp < 0

                    body_kp[:, 0] = (body_kp[:, 0] * (body_xmax - body_xmin) + body_xmin)
                    body_kp[:, 1] = (body_kp[:, 1] * (body_ymax - body_ymin) + body_ymin)
                    
                    face_kp[:, 0] = (face_kp[:, 0] * (body_xmax - body_xmin) + body_xmin) 
                    face_kp[:, 1] = (face_kp[:, 1] * (body_ymax - body_ymin) + body_ymin)

                    face_kp[invalid_face_kp] = -1
                    body_kp[invalid_body_kp] = -1

                    face_kp = face_kp.tolist()
                    body_kp = body_kp.tolist()

                    hand_kp = filtered_right_hand_kpts_list[i] + filtered_left_hand_kpts_list[i]

                    for kp in hand_kp:
                        if kp[0] > 0 and kp[1] > 0:
                            cv2.circle(frame_rgb, (int(kp[0]*ori_w), int(kp[1]*ori_h)), 1, (0, 255, 0), -1)
                    for kp in body_kp:
                        if kp[0] > 0 and kp[1] > 0:
                            cv2.circle(frame_rgb, (int(kp[0]), int(kp[1])), 1, (0, 255, 255), -1)
                    
                    for kp in face_kp:
                        if kp[0] > 0 and kp[1] > 0:
                            cv2.circle(frame_rgb, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), -1)

                    # print('len(face_kp):', len(face_kp))
                    # print('hand_kp:', hand_kp)
                    # print(f"    Saved frame {self.video_dir}/{frame_name}")
                    # cv2.imwrite(f'output/{frame_name}', frame_rgb[:,:,::-1])

                    # draw text on the image
                    cv2.putText(frame_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(f'{save_img_dir}/filtered_{frame_name}', frame_rgb[:,:,::-1])

            else:
                print(f"Filtered Sample Error: No frames sampled from {video_path}")
                continue
                    

            ## uniform sample from origin video_path and draw keypoints
            selected_frames, frame_index = self.uniform_sample(video_path, start_time, end_time, print_flag=False)
            if not selected_frames is None and len(selected_frames) > 0:
                print(f"Uniform Detected {len(filtered_frames)} / {len(filtered_frames)} frames, in {video_path}")
                for i, frame_bgr in enumerate(selected_frames):
                    frame_id = frame_index[i]
                    formated_frame_id = f'{frame_id:08d}'
                    frame_name = f'{video_name_prefix}_C{clip_id}_fid_{formated_frame_id}.jpg'

                    # draw text on the image
                    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imwrite(f'{save_img_dir}/uniform_{frame_name}', frame_bgr)
                    # print(f"    Saved frame {self.video_dir}/{frame_name}")
                    #            
            else:
                print(f"Uniform Sample Error: No frames sampled from {video_path}")
                
            available_cnt += 1
            if available_cnt > 5:
                break
            print()

        
                
def main_dump_anno_json(args):
    dataset = YouTubeASLPieces()
    print(f"Total clips in dataset: {len(dataset)}")
    
    # Example: Get the first item
    index = np.random.randint(0, len(dataset))
    print(f"Getting item at index {index}")
    item = dataset[index]
    if item:
        frames, text, keypoints = item
        print(f"Text: {text}")
        print(f"Frames shape: {frames.shape}")
        print(f"Hand keypoints (first frame): {keypoints['hand'].shape}")
        print(f"Body keypoints (first frame): {keypoints['body'].shape}")
        print(f"Face keypoints (first frame): {keypoints['face'].shape}")

        image = frames[0].permute(1, 2, 0).numpy()*255
        image = image.astype(np.uint8)
        h, w = image.shape[:2]

        hand_kps = keypoints['hand'][0]
        hand_kps[:, 0] *= w
        hand_kps[:, 1] *= h
        hand_kps = hand_kps.astype(int).tolist()
        
        body_kps = keypoints['body'][0]
        body_kps[:, 0] *= w
        body_kps[:, 1] *= h
        body_kps = body_kps.astype(int).tolist()
        
        face_kps = keypoints['face'][0]
        face_kps[:, 0] *= w
        face_kps[:, 1] *= h
        face_kps = face_kps.astype(int).tolist()


        # print('hand_kps:', hand_kps)
        # print('body_kps:', body_kps)
        # print('face_kps:', face_kps)    
        for (x, y) in hand_kps:  
            if x > 0 and y > 0:
                cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

        for (x, y) in body_kps:
            if x > 0 and y > 0:
                cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)
                    
        for (x, y) in face_kps:
            if x > 0 and y > 0:
                cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)


        cv2.imwrite('example.jpg', image)
        print("Saved example.jpg")

    save_img_dir = '/scratch/rhong5/dataset/youtubeASL_frames'
    save_anno_dir = '/scratch/rhong5/dataset/youtubeASL_anno'

    
#     save_img_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_frames'
#     save_anno_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno'


    dataset.dump_anno(save_img_dir, save_anno_dir, start_idx=args.start_idx, end_idx=args.end_idx)
    print("Dumped annotation data to JSON file.")
            
def main_sample_examples():
    dataset = YouTubeASLPieces(num_frames_per_clip = 15)
    dataset.sample_draw_kpts(save_img_dir = 'output2')

# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the dataset')
    parser.add_argument('--end_idx', type=int, default=None, help='End index of the dataset')
    args = parser.parse_args()
    # main_sample_examples()
    main_dump_anno_json(args)

# Example output: