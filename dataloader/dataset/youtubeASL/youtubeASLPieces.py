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
    parent_die = os.path.join(os.path.dirname(__file__), '../../..')
    sys.path.append(parent_die)
    

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping

class YouTubeASLPieces(Dataset):
    def __init__(self, video_dir: str, num_frames_per_clip: int = 12, target_img_size = (224, 224), debug = False):
        """
        Initialize the YouTube ASL dataset loader for a single video and VTT files.
        
        Args:
            video_dir (str): Directory containing the video ('video.mp4') and transcript (*.vtt) files.
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.video_dir = video_dir
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(f"Directory not found: {self.video_dir}")
        self.num_frames_per_clip = num_frames_per_clip
        self.target_img_size = target_img_size
        self.debug = debug
        self.keypoints_threshold = 0.1
        self.anno_file = '/scratch/rhong5/dataset/youtubeASL_annotation.txt'

        self.anno_data = self.parse_annotation(self.anno_file)
        
        print(f"Total clips in dataset: {len(self.anno_data)}")

        # # Find video and VTT files
        # self.video_transcript_pairs = self._find_files()
        
        #        
        # Initialize MediaPipe solutions
        # self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=self.keypoints_threshold)
        # self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)
        # self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)
        
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

    def _sample_frames_with_hand_filter(self, video_path: str, start_time: float, end_time: float) -> Optional[List[np.ndarray]]:
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
            print(f"Error opening video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # print(f"Video: {video_path}, FPS: {fps}, Total frames: {total_frames}")
        # print(f"Extracted segment: Start={start_time}, End={end_time}")


        # Convert times to frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if start_frame >= total_frames or end_frame > total_frames:

            print(f"Time range out of bounds for {video_path}")
            cap.release()
            return None

        end_frame = min(end_frame, total_frames - 1)
        total_clip_frames = end_frame - start_frame + 1

        if total_clip_frames <= 0:
            cap.release()
            return None

        # Uniformly sample N + 10 frames initially
        initial_num_frames = self.num_frames_per_clip + 10
        frame_indices = np.linspace(start_frame, end_frame, initial_num_frames, dtype=int)
        # frames = []
        hand_detected_frames = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect hands using MediaPipe
                results = self.mp_hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    ## pad frame_rgb to a square image based on the larger dimension
                    h, w, _ = frame_rgb.shape
                    if h > w:
                        pad = (h-w)//2
                        frame_rgb = cv2.copyMakeBorder(frame_rgb, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    elif w > h:
                        pad = (w-h)//2
                        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    hand_detected_frames.append(frame_rgb)
                # frames.append(frame_rgb)
            else:
                break

        cap.release()
        if not hand_detected_frames:
            return None

        # Uniformly sample num_frames from hand-detected frames
        if len(hand_detected_frames) <= self.num_frames_per_clip:
            sampled_frames = hand_detected_frames
        else:
            sample_indices = np.linspace(0, len(hand_detected_frames) - 1, self.num_frames_per_clip, dtype=int)
            sampled_frames = [hand_detected_frames[i] for i in sample_indices]

        return sampled_frames

    def _detect_keypoints(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """
        Detect hand, body, and face keypoints using MediaPipe.
        
        Args:
            frame (np.ndarray): RGB frame of shape (height, width, 3).
        
        Returns:
            Tuples of (hand_keypoints, body_keypoints, face_keypoints), each as lists of (x, y, confidence).
        """

        # Detect hands
        results_hands = self.mp_hands.process(frame)
        left_hand_keypoints, right_hand_keypoints = [(-1.0, -1.0)]*len(self.hand_indices), [(-1.0, -1.0)]*len(self.hand_indices)

        if results_hands.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness)):
                hand_label = handedness.classification[0].label  # "Left" or "Right"

                hand_keypoints = [(-1.0, -1.0)] * len(self.hand_indices)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx in self.hand_indices:  # 只选取 21 个关键点
                        pos = self.hand_indices.index(idx)  # Map to body indices (0–8)
                        x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                        # confidence = 1.0  # MediaPipe Hands 没有 visibility
                        hand_keypoints[pos] = (x, y)

                if hand_label == "Left":
                    left_hand_keypoints = hand_keypoints  # 确保左手存储
                else:
                    right_hand_keypoints = hand_keypoints  # 确保右手存储



        hand_keypoints = right_hand_keypoints + left_hand_keypoints


        # Detect body (pose)
        body_keypoints = [(-1.0, -1.0)] * len(self.body_indices)
        results_pose = self.mp_pose.process(frame)
        if results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if idx in self.body_indices:  # Use only specified body indices (0–8)
                    pos = self.body_indices.index(idx)
                    x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                    # confidence = landmark.visibility
                    body_keypoints[pos] = (x, y)



        # Detect face
        face_keypoints = [(-1.0, -1.0)] * len(self.face_indices)  # 预填充 18 个点
        results_face = self.mp_face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in self.face_indices:  # Use only specified face indices (0–17)
                        pos = self.face_indices.index(idx)
                        x, y = landmark.x * self.target_img_size[1], landmark.y * self.target_img_size[0]
                        # confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
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

        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=self.keypoints_threshold)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=self.keypoints_threshold)

        video_name, vtt_name, timestamp, text = self.anno_data[idx]

        start_time, end_time = timestamp.split(' --> ')
        # print(f"Processing video {video_name}, text: {text}, start: {start_time}, end: {end_time}")

        start_time = self._time_to_seconds(start_time)
        end_time = self._time_to_seconds(end_time)

        video_path = os.path.join(self.video_dir, video_name)
        
        
        # Sample frames with hand filtering
        sampled_frames = self._sample_frames_with_hand_filter(video_path, start_time, end_time)
        # print(f"Sampled frames: {len(sampled_frames)}, target frames: {self.num_frames_per_clip}")
    

        if sampled_frames is None or len(sampled_frames) == 0:
            return self.__getitem__(np.random.randint(0, len(self.anno_data)))

        # Process each frame for keypoints
        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        for frame in sampled_frames:
            hand_kp, body_kp, face_kp = self._detect_keypoints(frame)
            # print('len(face_kp):', len(face_kp))
            hand_keypoints_all.append(hand_kp)
            body_keypoints_all.append(body_kp)
            face_keypoints_all.append(face_kp)

        # print('face_keypoints_all:', face_keypoints_all)
        # print('face_keypoints_all:', face_keypoints_all)
        hand_keypoints_all = np.array(hand_keypoints_all) # (N, 42, 2)
        body_keypoints_all = np.array(body_keypoints_all) # (N, 9, 2)
        face_keypoints_all = np.array(face_keypoints_all) # (N, 18, 2)
        sampled_frames = np.array(sampled_frames) # (N, h, w, 3)

        pad_len = self.num_frames_per_clip - sampled_frames.shape[0]  # 计算填充长度

        if pad_len > 0:  # 仅当需要填充时执行
            sampled_frames = np.concatenate([sampled_frames, np.zeros((pad_len,) + sampled_frames.shape[1:])], axis=0)
            hand_keypoints_all = np.concatenate([hand_keypoints_all, np.full((pad_len, 2 * len(self.hand_indices), 2), -1)], axis=0)
            body_keypoints_all = np.concatenate([body_keypoints_all, np.full((pad_len, len(self.body_indices), 2), -1)], axis=0)
            face_keypoints_all = np.concatenate([face_keypoints_all, np.full((pad_len, len(self.face_indices), 2), -1)], axis=0)



        # Stack frames into a tensor
        frames_tensor = torch.from_numpy(sampled_frames).float() / 255.0  # 归一化到 [0, 1]
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

# Example usage
if __name__ == '__main__':
    video_dir = '/scratch/rhong5/dataset/youtubeASL/'  # Replace with your directory path (containing 'video.mp4' and '*.vtt')
    dataset = YouTubeASL(video_dir, num_frames_per_clip=16, debug=True)
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

        hand_kps = keypoints['hand'][0].astype(int).tolist()
        body_kps = keypoints['body'][0].astype(int).tolist()
        face_kps = keypoints['face'][0].astype(int).tolist()

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


# Example output: