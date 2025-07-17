import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
import json
import subprocess
import random

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype
])


class PoseEstimator:
    def __init__(self, yolo_model='yolov8n.pt', device='cpu'):
        # Auto select device
        self.device = device

        mmpose_config = 'mmpose_model/rtmpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
        mmpose_checkpoint = 'mmpose_model/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth'
                # === 使用默认 RTMPose WholeBody 模型 ===
        if not os.path.exists(mmpose_config) or not os.path.exists(mmpose_checkpoint):
            os.makedirs('mmpose_model/rtmpose', exist_ok=True)

            # 下载 config
            if not os.path.exists(mmpose_config):
                print("⬇️ Downloading RTMPose WholeBody config...")
                subprocess.run([
                    'wget',
                    'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py',
                    '-O', mmpose_config
                ], check=True)

            # 下载 checkpoint
            if not os.path.exists(mmpose_checkpoint):
                print("⬇️ Downloading RTMPose WholeBody checkpoint (~250MB)...")
                subprocess.run([
                    'wget',
                    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth',
                    '-O', mmpose_checkpoint
                ], check=True)

        # === 初始化 MMPose 模型 ==
        # =


        self.pose_model = init_model(mmpose_config, mmpose_checkpoint, device=self.device)


        # Load YOLO model
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)

    def extract_person_bboxes(self, image):
        results = self.yolo(image)[0]
        person_bboxes = []
        for box, cls in zip(results.boxes.xyxy.cpu(), results.boxes.cls.cpu()):
            if int(cls) == 0:  # class 0 = person
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                person_bboxes.append({'bbox': [x1, y1, w, h]})
        return person_bboxes

    def estimate_pose(self, image, person_bboxes):
        results = inference_topdown(
            self.pose_model, image, person_bboxes, bbox_format='xywh', dataset='WholeBody')
        return results

    def process_video(self, video_path, output_dir, save_vis=False):
        
        face_keypoints_all, hand_keypoints_all, body_keypoints_all, frame_names = self.process_mediepipe_anno(video_path)
        
        
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        video_basename = os.path.basename(video_path)
        video_basename_noext = os.path.splitext(video_basename)[0]
        
        frame_idx = 0

        while True:
            frame_name_noext = frame_names[frame_idx] 
            frame_name_noext = os.path.splitext(frame_name_noext)[0]
            
            
            success, frame = cap.read()
            if not success:
                break

            # Step 1: Detect person bbox
            bboxes = self.extract_person_bboxes(frame)

            # Step 2: Estimate keypoints
            poses = self.estimate_pose(frame, bboxes)

            # Step 3: Save keypoints
            # for i, person in enumerate(poses):
            #     kpts = person.pred_instances.keypoints  # shape: (133, 3)
            #     np.save(os.path.join(output_dir, f"frame_{frame_idx:05d}_p{i}.npy"), kpts)

            # Optional: Save visualized frame
            if save_vis:
                mmpose_vis_frame = self.pose_model.visualize(frame.copy(), poses)
                
                mediapipe_frame = frame.copy()
                for i, (face_kpts, hand_kpts, body_kpts) in enumerate(zip(face_keypoints_all[frame_idx], hand_keypoints_all[frame_idx], body_keypoints_all[frame_idx])):
                    # Draw face keypoints
                    for x, y in face_kpts:
                        cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
                    # Draw hand keypoints
                    for x, y in hand_kpts:
                        cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                    
                    # Draw body keypoints
                    for x, y in body_kpts:
                        cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                
                compose_frame = np.hstack((mmpose_vis_frame, mediapipe_frame))

                cv2.imwrite(os.path.join(output_dir, f"{video_basename_noext}_{frame_name_noext}.jpg"), compose_frame)

            frame_idx += 1

        cap.release()
        print(f"✅ Processed {frame_idx} frames from {video_path}")
        
    
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
        
        max_videos = 6 if len(video_list) > 6 else len(video_list)
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
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output results')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    video_dir = '/projects/kosecka/hongrui/dataset/how2sign/processed_how2sign/train/frames'
    
    estimator = PoseEstimator(device=device)
    estimator.process_multiple_videos(video_dir, args.output_dir)
    print("Pose estimation completed.")