import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
import webvtt

class YouTubeASLClip(Dataset):
    def __init__(self, clip_dir: str, num_frames_per_clip: int = 16, frame_sample_rate: int = 30):
        """
        Initialize the YouTube ASL dataset loader for pre-cropped clips.
        
        Args:
            clip_dir (str): Directory containing clip (.mp4) and transcript (.txt) files (e.g., 'clip_0.mp4', 'clip_0.txt').
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.clip_dir = clip_dir
        self.num_frames_per_clip = num_frames_per_clip
        self.frame_sample_rate = frame_sample_rate
        self.clips = self._find_clips()
        
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)
        
        # Load adjacency matrices (assuming they’re saved as .pth files or defined)
        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = list(range(21))  # 0–20 (matches MediaPipe hand landmarks directly)
        self.body_indices = [15, 13, 11, 16, 14, 12, 7, 8, 0]  # Map to MediaPipe pose landmarks (0–8)
        self.face_indices = [234, 454, 205, 425, 152, 4, 61, 291, 13, 37, 41, 14, 17, 55, 59, 18, 0, 17]  # Map to MediaPipe face mesh landmarks (0–17)

    def _find_clips(self) -> List[Tuple[str, str]]:
        """
        Find matching clip (.mp4) and transcript (.txt) files in the directory.
        
        Returns:
            List of tuples (clip_path, txt_path) with matching prefixes (e.g., 'clip_0.mp4', 'clip_0.txt').
        """
        clip_files = sorted([f for f in os.listdir(self.clip_dir) if f.endswith('.mp4')])
        clip_transcript_pairs = []
        
        for clip_file in clip_files:
            base_name = clip_file.replace('.mp4', '')
            txt_file = f"{base_name}.txt"
            txt_path = os.path.join(self.clip_dir, txt_file)
            if os.path.exists(txt_path):
                clip_path = os.path.join(self.clip_dir, clip_file)
                clip_transcript_pairs.append((clip_path, txt_path))
        
        return clip_transcript_pairs

    def _read_text(self, txt_path: str) -> str:
        """
        Read the transcript text from a .txt file.
        
        Args:
            txt_path (str): Path to the transcript .txt file.
        
        Returns:
            String of the transcript text.
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading TXT file {txt_path}: {e}")
            return ""

    def _sample_frames_with_hand_filter(self, clip_path: str, num_frames: int) -> Optional[List[np.ndarray]]:
        """
        Sample frames from a clip, filter out frames without hand keypoints, and resample.
        
        Args:
            clip_path (str): Path to the clip video file.
            num_frames (int): Target number of frames to sample.
        
        Returns:
            List of sampled frames or None if sampling fails.
        """
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"Error opening video {clip_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None

        # Uniformly sample N + 10 frames initially
        initial_num_frames = num_frames + 10
        frame_indices = np.linspace(0, total_frames - 1, initial_num_frames, dtype=int)
        frames = []
        hand_detected_frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect hands using MediaPipe
                results = self.mp_hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    hand_detected_frames.append(frame_rgb)
                frames.append(frame_rgb)
            else:
                break

        cap.release()
        if not hand_detected_frames:
            return None

        # Uniformly sample num_frames from hand-detected frames
        if len(hand_detected_frames) <= num_frames:
            sampled_frames = hand_detected_frames
        else:
            sample_indices = np.linspace(0, len(hand_detected_frames) - 1, num_frames, dtype=int)
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
        hand_keypoints, body_keypoints, face_keypoints = None, None, None

        # Detect hands
        results_hands = self.mp_hands.process(frame)
        if results_hands.multi_hand_landmarks:
            hand_keypoints = []
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx in self.hand_indices:  # Use only specified hand indices (0–20)
                        x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                        confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                        hand_keypoints.append((x, y, confidence))

        # Detect body (pose)
        results_pose = self.mp_pose.process(frame)
        if results_pose.pose_landmarks:
            body_keypoints = []
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if idx in self.body_indices:  # Use only specified body indices (0–8)
                    x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                    confidence = landmark.visibility
                    body_keypoints.append((x, y, confidence))

        # Detect face
        results_face = self.mp_face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            face_keypoints = []
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in self.face_indices:  # Use adjacency matrix for face
                        x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                        confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                        face_keypoints.append((x, y, confidence))

        return hand_keypoints, body_keypoints, face_keypoints

    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.clips)

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
        clip_path, txt_path = self.clips[idx]
        
        # Read text
        text = self._read_text(txt_path)
        if not text:
            return None  # Skip invalid clips

        # Sample frames with hand filtering
        sampled_frames = self._sample_frames_with_hand_filter(clip_path, self.num_frames_per_clip)
        if sampled_frames is None or len(sampled_frames) == 0:
            return None  # Skip invalid clips

        # Process each frame for keypoints
        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        for frame in sampled_frames:
            hand_kp, body_kp, face_kp = self._detect_keypoints(frame)
            hand_keypoints_all.append(hand_kp if hand_kp else [])
            body_keypoints_all.append(body_kp if body_kp else [])
            face_keypoints_all.append(face_kp if face_kp else [])

        # Stack frames into a tensor
        frames = [cv2.resize(frame, (224, 224)) for frame in sampled_frames]
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0)).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, 224, 224)

        # Create keypoints dictionary
        keypoints_dict = {
            'hand': hand_keypoints_all,
            'body': body_keypoints_all,
            'face': face_keypoints_all
        }

        return frames_tensor, text, keypoints_dict



class YouTubeASL(Dataset):
    def __init__(self, video_dir: str, num_frames_per_clip: int = 15, frame_sample_rate: int = 30):
        """
        Initialize the YouTube ASL dataset loader for a single video and VTT files.
        
        Args:
            video_dir (str): Directory containing the video ('video.mp4') and transcript (*.vtt) files.
            num_frames_per_clip (int): Number of frames to sample per clip (default: 16).
            frame_sample_rate (int): Frames per second to sample from the video (default: 30 FPS).
        """
        self.video_dir = video_dir
        self.num_frames_per_clip = num_frames_per_clip
        self.frame_sample_rate = frame_sample_rate
        
        # Find video and VTT files
        video_files = [f for f in os.listdir(video_dir) if f == 'video.mp4']
        vtt_files = [f for f in os.listdir(video_dir) if f.startswith('video.') and f.endswith('.vtt')]
        
        if not video_files or not vtt_files:
            raise ValueError(f"No video or VTT files found in {video_dir}")
        
        self.video_path = os.path.join(video_dir, video_files[0])
        self.vtt_path = os.path.join(video_dir, vtt_files[0])  # Use the first matching VTT file
        self.captions = self._parse_vtt(self.vtt_path)
        
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)
        
        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = list(range(21))  # 0–20 (matches MediaPipe hand landmarks directly)
        self.body_indices = [15, 13, 11, 16, 14, 12, 7, 8, 0]  # Map to MediaPipe pose landmarks (0–8)
        self.face_indices = [234, 454, 205, 425, 152, 4, 61, 291, 13, 37, 41, 14, 17, 55, 59, 18, 0, 17]  # Map to MediaPipe face mesh landmarks (0–17)

    def _find_files(self) -> List[Tuple[str, str]]:
        """
        Find matching video (.video) and transcript (.vtt) files in the directory.
        
        Returns:
            List of tuples (video_path, vtt_path) with matching prefixes (e.g., 'xx.video', 'xx..vtt').
        """
        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.video')]
        video_transcript_pairs = []
        
        for video_file in video_files:
            base_name = video_file.replace('.video', '')
            vtt_files = [f for f in os.listdir(self.video_dir) if f.startswith(base_name) and f.endswith('.vtt')]
            if vtt_files:
                video_path = os.path.join(self.video_dir, video_file)
                vtt_path = os.path.join(self.video_dir, vtt_files[0])  # Use the first matching VTT file
                video_transcript_pairs.append((video_path, vtt_path))
        
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

    def _sample_frames_with_hand_filter(self, start_time: float, end_time: float) -> Optional[List[np.ndarray]]:
        """
        Sample frames from a video clip, filter out frames without hand keypoints, and resample.
        
        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
        
        Returns:
            List of sampled frames or None if sampling fails.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video {self.video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Convert times to frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if start_frame >= total_frames or end_frame > total_frames:
            print(f"Time range out of bounds for {self.video_path}")
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
        frames = []
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
                    hand_detected_frames.append(frame_rgb)
                frames.append(frame_rgb)
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
        hand_keypoints, body_keypoints, face_keypoints = None, None, None

        # Detect hands
        results_hands = self.mp_hands.process(frame)
        if results_hands.multi_hand_landmarks:
            hand_keypoints = []
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx in self.hand_indices:  # Use only specified hand indices (0–20)
                        x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                        confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                        hand_keypoints.append((x, y, confidence))

        # Detect body (pose)
        results_pose = self.mp_pose.process(frame)
        if results_pose.pose_landmarks:
            body_keypoints = []
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if idx in self.body_indices:  # Use only specified body indices (0–8)
                    x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                    confidence = landmark.visibility
                    body_keypoints.append((x, y, confidence))

        # Detect face
        results_face = self.mp_face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            face_keypoints = []
            for face_landmarks in results_face.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in self.face_indices:  # Use only specified face indices (0–17)
                        x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                        confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                        face_keypoints.append((x, y, confidence))

        return hand_keypoints, body_keypoints, face_keypoints

    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.captions)

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
        start_time, end_time, text = self.captions[idx]
        
        # Sample frames with hand filtering
        sampled_frames = self._sample_frames_with_hand_filter(start_time, end_time)
        if sampled_frames is None or len(sampled_frames) == 0:
            return None  # Skip invalid clips

        # Process each frame for keypoints
        hand_keypoints_all, body_keypoints_all, face_keypoints_all = [], [], []
        for frame in sampled_frames:
            hand_kp, body_kp, face_kp = self._detect_keypoints(frame)
            hand_keypoints_all.append(hand_kp if hand_kp else [])
            body_keypoints_all.append(body_kp if body_kp else [])
            face_keypoints_all.append(face_kp if face_kp else [])

        # Stack frames into a tensor
        frames = [cv2.resize(frame, (224, 224)) for frame in sampled_frames]
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0)).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, 224, 224)

        # Create keypoints dictionary
        keypoints_dict = {
            'hand': hand_keypoints_all,
            'body': body_keypoints_all,
            'face': face_keypoints_all
        }

        return frames_tensor, text, keypoints_dict

# Example usage
def main_YouTubeASL():
    video_dir = "path/to/your/directory"  # Replace with your directory path (containing 'video.mp4' and '*.vtt')
    dataset = YouTubeASL(video_dir, num_frames_per_clip=16, frame_sample_rate=30)
    print(f"Total clips in dataset: {len(dataset)}")
    
    # Example: Get the first item
    item = dataset[0]
    if item:
        frames, text, keypoints = item
        print(f"Text: {text}")
        print(f"Frames shape: {frames.shape}")
        print(f"Hand keypoints (first frame): {keypoints['hand'][0][:5] if keypoints['hand'][0] else 'None'}")
        print(f"Body keypoints (first frame): {keypoints['body'][0][:5] if keypoints['body'][0] else 'None'}")
        print(f"Face keypoints (first frame): {keypoints['face'][0][:5] if keypoints['face'][0] else 'None'}")

# Example usage
def main_YouTubeASLClip():
    clip_dir = "path/to/your/clips"  # Replace with your directory path
    dataset = YouTubeASL(clip_dir, num_frames_per_clip=16, frame_sample_rate=30)
    print(f"Total clips in dataset: {len(dataset)}")
    
    # Example: Get the first item
    item = dataset[0]
    if item:
        frames, text, keypoints = item
        print(f"Text: {text}")
        print(f"Frames shape: {frames.shape}")
        print(f"Hand keypoints (first frame): {keypoints['hand'][0][:5] if keypoints['hand'][0] else 'None'}")
        print(f"Body keypoints (first frame): {keypoints['body'][0][:5] if keypoints['body'][0] else 'None'}")
        print(f"Face keypoints (first frame): {keypoints['face'][0][:5] if keypoints['face'][0] else 'None'}")