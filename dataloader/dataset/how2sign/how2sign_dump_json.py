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

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_dir)
    

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping


def filter_out_to_do_list():
    already_processed_list = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno_all_filepaths.txt'
    to_be_processed_list = 'to_be_processed_clips.txt'
    ## read already processed list
    processed_clip_ids = []
    if os.path.exists(already_processed_list):
        with open(already_processed_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                filename = os.path.basename(line)  # youtubeASL_Clip00000047_anno.json
                # 提取中间的数字部分
                processed_id = filename.split('_')[1].replace('Clip', '')  # '00000047'
                processed_clip_ids.append(int(processed_id))

    processed_clip_ids = set(processed_clip_ids)  # 去重
    to_be_processed_ids = set(range(0, 652092)) - processed_clip_ids  # youtube-asl总共有 652092 个 clip_id
    ## read to be processed list
    with open(to_be_processed_list, 'w') as f:
        ## write clip_ids to to_be_processed_list.txt
        for clip_id in to_be_processed_ids:
            f.write(f"{clip_id}\n")



class how2signDumpJson:
    def __init__(self, root_dir: str = None, split = 'test', num_frames_per_clip: int = 250, debug = False, return_frame = False):
        """
        Initialize the YouTube ASL dataset loader for a single video and VTT files.
        
        Args:
            video_dir (str): Directory containing the video ('video.mp4') and transcript (*.vtt) files.
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        if split == 'train':
            self.root_dir = root_dir if root_dir is not None else '/projects/kosecka/hongrui/dataset/how2sign/video_zipfiles/'
        elif split == 'test':
            self.root_dir = '/scratch/rhong5/dataset/how2sign/video_level/test/rgb_front'
        # self.video_dir = os.path.join(f'{self.root_dir}', f'video_level/{split}/rgb_front/raw_videos') 
        self.video_dir = os.path.join(f'{self.root_dir}', f'raw_videos') 
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
        
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        
        self.num_frames_per_clip = num_frames_per_clip

        self.debug = debug
        self.return_frame = return_frame  # 是否返回frame

        print('loading YOLOv8 model...')
        self.yolo_model = YOLO('/home/rhong5/research_pro/hand_modeling_pro/signLanguageTrans/dataloader/dataset/youtubeASL/yolov8n.pt')
        
        # self.yolo_model.fuse()  # Fuse model for faster inference

        print('finished loading YOLOv8 model')
        self.keypoints_threshold = 0.35  # Confidence threshold for keypoint detection
        self.anno_file = f'/projects/kosecka/hongrui/dataset/how2sign/re-aligned_how2sign_realigned_{split}.txt'
        
        self.to_be_processed_list = f'{split}_to_be_processed_clips.txt'
        
        self.anno_info_lists = self.parse_annotation(self.anno_file)
        print(f"Total clips in dataset: {len(self.anno_info_lists)}")
        
        if os.path.exists(self.to_be_processed_list):
            with open(self.to_be_processed_list, 'r') as f:
                self.to_be_processed_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        else:
            self.to_be_processed_ids = list(range(0, len(self.anno_info_lists)))  # YouTube ASL has 652092 clips
            
        print(f"Total clips to be processed: {len(self.to_be_processed_ids)}")
        
    

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
        lines = []
        with open(anno_file, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split("\t")  # 去除换行并按 tab 分割
                lines.append(fields)

        # 示例输出：
        # [
        #   ['-g0iPSnQt6w-1-rgb_front', '01', '7.97', '13.83', "I'm an expert on diving, talking about a back 1 1/2 pike."],
        #   ...
        # ]

        return lines

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



    def _time_to_seconds(self, time_str: str) -> float:
        """
        Convert VTT time string (e.g., "00:00:10.052") to seconds.
        
        Args:
            time_str (str): Time string in format "HH:MM:SS.mmm".
        
        Returns:
            Float representing seconds.
        """
        time_senconds = float(time_str)
        return time_senconds


    def _detect_keypoints(self, frame_rgb: np.ndarray, frame_id: str = None, det_hand_kpts: bool = True) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """
        Detect body, and face keypoints using MediaPipe.
        
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



    def iou_filter(self, left_hand_bboxes, right_hand_bboxes, threshold=0.85):
        '''
        使用 IoU (Intersection over Union) 策略过滤帧，保留关键帧。
        只要有一个手的 IoU 小于等于 threshold，就保留该帧。
        Args:
            left_hand_bboxes (List[List[float]]): 左手 bounding box 列表
            right_hand_bboxes (List[List[float]]): 右手 bounding box 列表
            threshold (float): IoU 阈值，默认 0.85
        '''

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
        
        # filtered_frames = [selected_frames[0]]  # 存储最终的关键帧
        filtered_left_bboxes = [left_hand_bboxes[0]]
        filtered_right_bboxes = [right_hand_bboxes[0]]

        indices = [0]  # 存储最终的关键帧索引
        cur_idx = 0  # 当前帧索引

        for i in range(1, len(left_hand_bboxes)-2):  # 遍历后续帧
            left_iou = compute_iou(left_hand_bboxes[cur_idx], left_hand_bboxes[i])
            right_iou = compute_iou(right_hand_bboxes[cur_idx], right_hand_bboxes[i])

            if left_iou < threshold or right_iou < threshold:  
                # 只要有一个手的 IoU 小于等于 0.7，就保留该帧
                # filtered_frames.append(selected_frames[i])
                
                filtered_left_bboxes.append(left_hand_bboxes[i])
                filtered_right_bboxes.append(right_hand_bboxes[i])
                cur_idx = i  # 更新当前帧索引
                indices.append(i)
        return indices

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
        initial_num_frames = self.num_frames_per_clip + 20 if self.num_frames_per_clip + 20 < total_clip_frames else total_clip_frames
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

        max_det_frames = len(frame_indices)//3
        
        cnt = 0
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
                
                # 若多人，跳过 clip
                if len(person_bboxes) != 1:
                    cnt += 10
                    if print_flag:
                        print(f"Skip {video_path}: detected {len(person_bboxes)} persons.")
                    
                    if cnt > max_det_frames:
                        cap.release()
                        return None, None, None, None, None
                    continue
                
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
                        return None, None, None, None, None
                    continue

                body_bbox = [body_xmin, body_ymin, body_xmax, body_ymax]
                det_body_bbox = False
                
            ori_h, ori_w, _ = frame.shape
            left_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices)
            right_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print('body_bbox:', body_bbox)
            
            body_rgb = frame_rgb[body_ymin:body_ymax, body_xmin:body_xmax].copy()

            # Detect hands using MediaPipe
            results_hands = self.mp_hands.process(body_rgb)
            if results_hands.multi_hand_landmarks:
                # print(f"Detected hands in frame {idx}, landmarks: {len(results.multi_hand_landmarks)}")
                # print(f"Hand landmarks: {results.multi_hand_landmarks}")
                ## pad frame_rgb to a square image based on the larger dimension
                # if h > w:
                #     pad = (h-w)//2
                #     frame_rgb = cv2.copyMakeBorder(frame_rgb, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # elif w > h:
                #     pad = (w-h)//2
                #     frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                                
                body_h, body_w = body_rgb.shape[:2]
                
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)):
                    hand_label = handedness.classification[0].label  # "Left" or "Right"

                    hand_keypoints = [(-1.0, -1.0)] * len(self.hand_indices)
                    for ix, landmark in enumerate(hand_landmarks.landmark):
                        if ix in self.hand_indices:  # 只选取 21 个关键点
                            # print('idx:', idx, 'self.hand_indices:', self.hand_indices)
                            pos = self.hand_indices.index(ix)  # Map to body indices (0–8)
                            # x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                            # x, y = landmark.x, landmark.y
                            x = (landmark.x * body_w  + body_xmin ) / ori_w
                            y = (landmark.y * body_h + body_ymin) / ori_h
                            
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
                
                selected_frames.append(frame_rgb)
                frame_index.append(fid)                    
                left_hand_kpts_list.append(left_hand_keypoints)
                right_hand_kpts_list.append(right_hand_keypoints)
                left_hand_bboxes.append([left_xmin, left_ymin, left_xmax, left_ymax])
                right_hand_bboxes.append([right_xmin, right_ymin, right_xmax, right_ymax])
            
            cnt += 1
            
        cap.release()

        # print(f'frame_index:', frame_index)
        if not selected_frames:
            if print_flag:
                print(f"No hand-detected frames in {video_path}") 
            return None, None, None, None, None
        
        # Uniformly sample num_frames from hand-detected frames
        # if len(selected_frames) <= self.num_frames_per_clip:
        #     filtered_frames = selected_frames
        #     filtered_frames_index = frame_index
        #     filtered_left_hand_kpts_list = left_hand_kpts_list
        #     filtered_right_hand_kpts_list = right_hand_kpts_list
        # else:
        filtered_indices = self.iou_filter(left_hand_bboxes, right_hand_bboxes) # the indices of frames to keep in this clip, start from 0
        if print_flag:
            print(f"filtered {len(filtered_indices)} / detected {len(selected_frames)} / total {total_clip_frames} frames, in {video_path}")
            
        if len(filtered_indices) > self.num_frames_per_clip:
            ## uniform sample
            filtered_indices = self.greedy_sparse_sampling(filtered_indices, self.num_frames_per_clip)

        filtered_frames = [selected_frames[i] for i in filtered_indices]
        filtered_frames_index = [frame_index[i] for i in filtered_indices]
        filtered_left_hand_kpts_list = [left_hand_kpts_list[i] for i in filtered_indices]
        filtered_right_hand_kpts_list = [right_hand_kpts_list[i] for i in filtered_indices]

        
        return filtered_frames, filtered_frames_index, [filtered_left_hand_kpts_list, filtered_right_hand_kpts_list], body_bbox, total_clip_frames
    
    def greedy_sparse_sampling(self, indices, target_num):
        indices = sorted(indices)
        if target_num >= len(indices):
            return indices

        selected = [indices[0]]
        remaining = set(indices[1:])

        while len(selected) < target_num and remaining:
            # 贪心选择“距离当前 selected 最远”的点
            best_idx = max(
                remaining,
                key=lambda x: min(abs(x - s) for s in selected)
            )
            selected.append(best_idx)
            remaining.remove(best_idx)

        return sorted(selected)

    
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
        cnt_clip = 0
        
        to_be_processed_ids = self.to_be_processed_ids[start_idx:end_idx]
        
        for i, idx in enumerate(to_be_processed_ids):
            clip_info_dict = {}
            clip_id = f'{idx:08d}'

            video_name_prefix, scentence_id, start_time, end_time, text = self.anno_info_lists[idx]
            video_name = f'{video_name_prefix}.mp4'

            print(f"Processing clip {video_name} {i + start_idx}/{end_idx}")


            # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")

            start_time = self._time_to_seconds(start_time)
            end_time = self._time_to_seconds(end_time)

            video_path = os.path.join(self.video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"Warning: video {video_name} {video_path} does not exist.")
                continue
            
                
            if text.endswith('.'):
                drop_last = 5
            else:
                drop_last = 3

            
            # Sample frames with hand filtering
            filtered_frames, filtered_frames_index, \
            filtered_hand_kpts_list, body_bbox, original_frame_num = self.fine_sample_frames_with_filter(video_path, start_time, 
                                                                                           end_time, drop_last, print_flag=True)

            if filtered_frames is None or len(filtered_frames) == 0:
                print(f"Warning: No valid frames found in {video_name} for clip {clip_id}.")
                continue
            
            filtered_left_hand_kpts_list, filtered_right_hand_kpts_list = filtered_hand_kpts_list
            # print(f"Sampled frames: {len(filtered_frames)}, {filtered_frames_index}")
        
            
            height, width, _ = filtered_frames[0].shape

            clip_info_dict['video_name'] = video_name
            clip_info_dict['clip_id'] = clip_id
            clip_info_dict['text'] = text
            clip_info_dict['start_time'] = start_time
            clip_info_dict['end_time'] = end_time
            clip_info_dict['height'] = height
            clip_info_dict['width'] = width
            clip_info_dict['original_frame_num'] = original_frame_num
            clip_info_dict['filtered_frame_num'] = len(filtered_frames)
            clip_info_dict['keyframes'] = {}
            
            body_xmin, body_ymin, body_xmax, body_ymax = body_bbox
            # Process each frame for keypoints
            new_video_frames = []
            for i, frame_rgb in enumerate(filtered_frames):
                frame_info_dict = {}
                frame_id = filtered_frames_index[i]
                formated_frame_id = f'{frame_id:06d}'
                new_frame_id = f'{i:06d}'
                
                frame_name = new_frame_id
                
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
                # cv2.imwrite(f'{save_img_dir}/{frame_name}', frame_rgb[:,:,::-1])
                new_video_frames.append(frame_rgb[:,:,::-1])  # Convert RGB to BGR for saving
                
                # print(f"    Saved frame {self.video_dir}/{frame_name}")
                # cv2.imwrite(f'output/{frame_name}', frame_rgb[:,:,::-1])
                frame_info_dict['hand'] = hand_kp
                frame_info_dict['body'] = body_kp
                frame_info_dict['face'] = face_kp
                frame_info_dict['original_frame_id'] = formated_frame_id
                # frame_info_dict['frame_name'] = frame_name
                clip_info_dict['keyframes'][frame_name] = frame_info_dict
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
                
            height, width, _ = new_video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 3
            output_path = os.path.join(save_img_dir,  f'{video_name_prefix}_SID{scentence_id}_frames.mp4')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in new_video_frames:
                writer.write(frame)  # 确保 frame 是 BGR 格式
            writer.release()
            print(f"✅ 成功保存视频到: {output_path}")


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
        json_file = os.listdir(anno_json_dir)[0]  # Assuming only one JSON file in the directory
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
        for frame_name, frame_info in anno_data['keyframes'].items():
            
            #
            frame_id = int(frame_name)
            new_frame_name = f'{frame_name}.jpg'
            frame = frames[frame_id]
            # print('frame_rgb.shape:', frame_rgb.shape)
            # print(f'frame_rgb.dtype:', frame_rgb.dtype)
            hand_kp = frame_info['hand']
            body_kp = frame_info['body']
            face_kp = frame_info['face']
            h, w, _ = frame.shape
            frame = frame.copy()
            # print('hand_kp:', hand_kp)
            for (x, y) in hand_kp:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)
            for (x, y) in body_kp:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x * w), int(y * h)), 1, (255, 0, 0), -1)
            for (x, y) in face_kp:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 0, 255), -1)
            
            cv2.imwrite(f'{output_dir}/{new_frame_name}', frame)
            cnt += 1
            if cnt > 5:
                break

                
        
        
                
def main_dump_anno_json(args):
    debug = args.debug
    split = args.split if hasattr(args, 'split') else 'train'
    print(f"Running in debug mode: {debug}, split: {split}")
    
    

    dataset = how2signDumpJson(split = split, debug=debug)

    
    if debug:
        save_img_dir = f'temp/{split}/frames'
        save_anno_dir = f'temp/{split}/annos'
        output_dir = 'temp/output2'
        os.makedirs(output_dir, exist_ok=True)
    else:
        # save_img_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_frames'
        # save_anno_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno'
        # save_img_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_frames'
        # save_anno_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_anno'
        save_img_dir = f'/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/{split}/frames'
        save_anno_dir = f'/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/{split}/annos'

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