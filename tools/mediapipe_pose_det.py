import os
import cv2
import torch
import json
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import random


class PoseEstimator:
    def __init__(self, yolo_model='yolov8n.pt', device='cpu'):
        self.device = device
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True
        )

    def det_person_bboxes(self, image):
        results = self.yolo(image)[0]
        person_bboxes = []
        for box, cls in zip(results.boxes.xyxy.cpu(), results.boxes.cls.cpu()):
            if int(cls) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.tolist())
                person_bboxes.append({'bbox': [x1, y1, x2, y2]})
        return person_bboxes

    def detect_landmarks(self, image):
        """Detect holistic landmarks using MediaPipe."""
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output = {
            'pose_landmarks': [],
            'face_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': []
        }

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                output['pose_landmarks'].append([lm.x, lm.y, lm.z, lm.visibility])

        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                output['face_landmarks'].append([lm.x, lm.y, lm.z])

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                output['left_hand_landmarks'].append([lm.x, lm.y, lm.z])

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                output['right_hand_landmarks'].append([lm.x, lm.y, lm.z])

        return output, results  # also return raw results for visualization

    def process_video(self, video_path, output_dir, save_vis=False):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        face_keypoints_all, hand_keypoints_all, body_keypoints_all, frame_names = self.process_mediepipe_anno(video_path)

        frame_idx = 0
        video_basename = os.path.basename(video_path)
        video_basename_noext = os.path.splitext(video_basename)[0]


        all_results = []

        while frame_idx < len(frame_names):
            success, frame = cap.read()
            if not success:
                break
            frame_name_noext = frame_names[frame_idx]
            # Step 1: Detect person bboxes using YOLO
            bboxes = self.det_person_bboxes(frame.copy())

            person_bbox = bboxes[0]['bbox']
            x1, y1, x2, y2 = person_bbox
            cropped = frame[y1:y2, x1:x2]

            landmarks, raw_results = self.detect_landmarks(cropped)

            if save_vis:
                mmpose_vis_frame = frame.copy()
                annotated = cropped.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, raw_results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                # mp.solutions.drawing_utils.draw_landmarks(
                #     annotated, raw_results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, raw_results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, raw_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

                
                mediapipe_frame = frame.copy()
                mediapipe_frame = frame.copy()
                face_kpts = face_keypoints_all[frame_idx]
                hand_kpts = hand_keypoints_all[frame_idx]
                body_kpts = body_keypoints_all[frame_idx]
                # Draw face keypoints
                for x, y in face_kpts:
                    cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Draw hand keypoints
                for x, y in hand_kpts:
                    cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                
                # Draw body keypoints
                for x, y in body_kpts:
                    cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                
                mmpose_vis_frame[y1:y2, x1:x2] = annotated
                compose_frame = np.hstack((mmpose_vis_frame, mediapipe_frame))

                cv2.imwrite(os.path.join(output_dir, f"{video_basename_noext}_{frame_name_noext}.jpg"), compose_frame)

                frame_idx += 1

            frame_idx += 1

        cap.release()
        # self.holistic.close()

        # Save results


    def process_mediepipe_anno(self, video_path):        

        json_filepath = video_path.replace('_frames.mp4','_anno.json')
        json_filepath = json_filepath.replace('/frames/','/annos/')
        
        
        with open(json_filepath, 'r', encoding='utf-8') as f:
            clip_anno_info = json.load(f)

        # Read text
        text = clip_anno_info['text']
        
        keyframes_dict = clip_anno_info['keyframes']
        
        frame_names = list(keyframes_dict.keys())

        width = clip_anno_info['width']
        height = clip_anno_info['height']

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
        
        ### get the bbox of the body
        body_keypoints_all[:, :, 0] *= width
        body_keypoints_all[:, :, 1] *= height
        body_keypoints_all = body_keypoints_all.astype(int)
        
        hand_keypoints_all[:, :, 0] *= width
        hand_keypoints_all[:, :, 1] *= height
        hand_keypoints_all = hand_keypoints_all.astype(int)
        
        face_keypoints_all[:, :, 0] *= width
        face_keypoints_all[:, :, 1] *= height
        face_keypoints_all = face_keypoints_all.astype(int)
        
        
        return face_keypoints_all, hand_keypoints_all, body_keypoints_all, frame_names
        

    def process_multiple_videos(self, video_dir, output_root, save_vis=True):
        video_list = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        random.shuffle(video_list)  # Shuffle the video list for randomness
        
        max_videos = 3 if len(video_list) > 6 else len(video_list)
        video_list = video_list[:max_videos]
        for video_path in video_list:
            name = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(output_root, name)
            self.process_video(video_path, out_dir, save_vis=save_vis)
            
        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pose Estimation with YOLO and MMPose")
    parser.add_argument('--mmpose_config', type=str, default=None, help='Path to MMPose config file')
    parser.add_argument('--mmpose_checkpoint', type=str, default=None, help='Path to MMPose checkpoint file')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='Path to YOLO model file')
    
    parser.add_argument('--video_dir', type=str, default='', help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='output2', help='Directory to save output results')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    video_dir = '/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/train/frames'
    
    estimator = PoseEstimator(device=device)
    estimator.process_multiple_videos(video_dir, args.output_dir)
    print("Pose estimation completed.")