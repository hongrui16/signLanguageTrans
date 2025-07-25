# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from typing import List, Tuple

import cv2

import loguru
import numpy as np
import onnxruntime as ort
import torch
import os
from ultralytics import YOLO
import json
import random


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["KMP_AFFINITY"] = "none"

logger = loguru.logger




def preprocess(
    img: np.ndarray, input_size: Tuple[int, int] = (288, 384), bbox = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do preprocessing for RTMPose model inference.

    Args:
        img (np.ndarray): Input image in shape.
        input_size (tuple): Input image size in shape (w, h).

    Returns:
        tuple:
        - resized_img (np.ndarray): Preprocessed image.
        - center (np.ndarray): Center of image.
        - scale (np.ndarray): Scale of image.
    """
    # get shape of image
    img_shape = img.shape[:2]
    if bbox is None:
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    # get center and scale
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)

    # do affine transformation
    resized_img, scale = top_down_affine(input_size, scale, center, img)

    # normalize image
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    resized_img = (resized_img - mean) / std

    return resized_img, center, scale


def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    """Build onnxruntime session.

    Args:
        onnx_file (str): ONNX file path.
        device (str): Device type for inference.

    Returns:
        sess (ort.InferenceSession): ONNXRuntime session.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1


    providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, sess_options=so, providers=providers)

    return sess


def inference(sess: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    """Inference RTMPose model.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        img (np.ndarray): Input image in shape.

    Returns:
        outputs (np.ndarray): Output of RTMPose model.
    """
    # build input
    input = [img.transpose(2, 0, 1)]

    # build output
    sess_input = {sess.get_inputs()[0].name: input}
    sess_output = []
    for out in sess.get_outputs():
        sess_output.append(out.name)

    # run model
    outputs = sess.run(sess_output, sess_input)

    return outputs


def postprocess(outputs: List[np.ndarray],
                model_input_size: Tuple[int, int],
                center: Tuple[int, int],
                scale: Tuple[int, int],
                simcc_split_ratio: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess for RTMPose model output.

    Args:
        outputs (np.ndarray): Output of RTMPose model.
        model_input_size (tuple): RTMPose model Input image size.
        center (tuple): Center of bbox in shape (x, y).
        scale (tuple): Scale of bbox in shape (w, h).
        simcc_split_ratio (float): Split ratio of simcc.

    Returns:
        tuple:
        - keypoints (np.ndarray): Rescaled keypoints.
        - scores (np.ndarray): Model predict scores.
    """
    # use simcc to decode
    simcc_x, simcc_y = outputs
    keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

    # rescale keypoints
    keypoints = keypoints / model_input_size * scale + center - scale / 2

    return keypoints, scores


def visualize(img: np.ndarray,
              keypoints: np.ndarray,
              scores: np.ndarray,
              filename: str = 'output.jpg',
              tag = 'mpose',
              thr=0.3) -> np.ndarray:
    """Visualize the keypoints and skeleton on image.

    Args:
        img (np.ndarray): Input image in shape.
        keypoints (np.ndarray): Keypoints in image.
        scores (np.ndarray): Model predict scores.
        thr (float): Threshold for visualize.

    Returns:
        img (np.ndarray): Visualized image.
    """
    # default color
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                (129, 130), (130, 131), (131, 132)]
    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    link_color = [
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]
    point_color = [
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
        4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    # draw keypoints and skeleton
    for kpts, score in zip(keypoints, scores):
        keypoints_num = len(score)
        for kpt, color in zip(kpts, point_color):
            cv2.circle(img, tuple(kpt.astype(np.int32)), 1, palette[color], 1,
                       cv2.LINE_AA)
        for (u, v), color in zip(skeleton, link_color):
            if u < keypoints_num and v < keypoints_num \
                        and score[u] > thr and score[v] > thr:
                cv2.line(img, tuple(kpts[u].astype(np.int32)),
                         tuple(kpts[v].astype(np.int32)), palette[color], 2,
                         cv2.LINE_AA)

    # save to local
    # cv2.imwrite(filename, img)
    img = put_tag_on_image(img, tag)
    
    return img

def put_tag_on_image(img: np.ndarray, tag: str = 'mpose') -> np.ndarray:
    """Put a tag on the image.

    Args:
        img (np.ndarray): Input image in shape.
        tag (str): Tag to put on the image.

    Returns:
        img (np.ndarray): Image with tag.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, tag, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return img

def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (int): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

    

class PoseEstimator:
    def __init__(self, yolo_model='yolov8n.pt', device='cpu'):
        # Auto select device
        self.device = device

        # Load YOLO model
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        onnx_file = 'rtmpose_onnx/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728/end2end.onnx'
    
        logger.info('Start running model on RTMPose...')

        if torch.cuda.is_available():
            device = 'cuda'
            logger.info('Using CUDA for inference...')
        else:
            device = 'cpu'
            logger.info('Using CPU for inference...')


        # build onnx model
        logger.info('2. Build onnx model from {}...'.format(onnx_file))
        self.sess = build_session(onnx_file, device)
        h, w = self.sess.get_inputs()[0].shape[2:]
        self.model_input_size = (w, h)

    def det_person_bboxes(self, image):
        results = self.yolo(image)[0]
        person_bboxes = []
        for box, cls in zip(results.boxes.xyxy.cpu(), results.boxes.cls.cpu()):
            if int(cls) == 0:  # class 0 = person
                x1, y1, x2, y2 = box.tolist()
                # w, h = x2 - x1, y2 - y1
                person_bboxes.append({'bbox': [x1, y1, x2, y2]})
        return person_bboxes

    def det_pose(self, image, person_bboxes):
            
        # preprocessing
        logger.info('3. Preprocess image...')
        resized_img, center, scale = preprocess(image, self.model_input_size)

        # inference
        logger.info('4. Inference...')
        start_time = time.time()
        outputs = inference(self.sess, resized_img)
        end_time = time.time()
        logger.info('4. Inference done, time cost: {:.4f}s'.format(end_time -
                                                                start_time))

        # postprocessing
        logger.info('5. Postprocess...')
        keypoints, scores = postprocess(outputs, self.model_input_size, center, scale)
        return keypoints, scores

    def process_video(self, video_path, output_dir, save_vis=False):
        
        face_keypoints_all, hand_keypoints_all, body_keypoints_all, frame_names = self.process_mediepipe_anno(video_path)
        # print('face_keypoints_all', face_keypoints_all.shape
        #       , 'hand_keypoints_all', hand_keypoints_all.shape,
        #       'body_keypoints_all', body_keypoints_all.shape)
        
        # print('frame_names', frame_names)
        
        
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        video_basename = os.path.basename(video_path)
        video_basename_noext = os.path.splitext(video_basename)[0]
        
        frame_idx = 0

        while frame_idx < len(frame_names):
            frame_name_noext = frame_names[frame_idx] 
            # frame_name_noext = os.path.splitext(frame_name_noext)[0]
            
            
            success, frame = cap.read()
            
            if not success:
                break

            # Step 1: Detect person bbox
            bboxes = self.det_person_bboxes(frame.copy())

            # Step 2: Estimate keypoints
            poses, scores = self.det_pose(frame.copy(), bboxes)

            # Step 3: Save keypoints
            # for i, person in enumerate(poses):
            #     kpts = person.pred_instances.keypoints  # shape: (133, 3)
            #     np.save(os.path.join(output_dir, f"frame_{frame_idx:05d}_p{i}.npy"), kpts)

            # Optional: Save visualized frame
            if save_vis:
                # logger.info('6. Visualize inference result...')
                mmpose_vis_frame = visualize(frame.copy(), poses, scores, output_dir)

                # print('face_keypoints_all', face_keypoints_all[frame_idx].shape
                #       , 'hand_keypoints_all', hand_keypoints_all[frame_idx].shape,
                #       'body_keypoints_all', body_keypoints_all[frame_idx].shape)
                
                mediapipe_frame = frame.copy()
                face_kpts = face_keypoints_all[frame_idx]
                hand_kpts = hand_keypoints_all[frame_idx]
                body_kpts = body_keypoints_all[frame_idx]
                # print('face_kpts', face_kpts)
                # Draw face keypoints
                for x, y in face_kpts.tolist():
                    cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Draw hand keypoints
                for x, y in hand_kpts.tolist():
                    cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                
                # Draw body keypoints
                for x, y in body_kpts.tolist():
                    cv2.circle(mediapipe_frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
                mediapipe_frame = put_tag_on_image(mediapipe_frame, 'mediapipe')

                
                compose_frame = np.hstack((mmpose_vis_frame, mediapipe_frame))
                out_img_filepath = os.path.join(output_dir, f"{video_basename_noext}_{frame_name_noext}.jpg")
                logger.info(f'Saving visualized frame to {out_img_filepath}')
                cv2.imwrite(out_img_filepath, compose_frame)

            frame_idx += 1

        cap.release()
        print(f"✅ Processed {frame_idx} frames from {video_path}")
        
    
    def process_mediepipe_anno(self, video_path):        

        json_filepath = video_path.replace('_frames.mp4','_anno.json')
        json_filepath = json_filepath.replace('/frames/','/annos/')

        print(json_filepath)
        if not os.path.exists(json_filepath):
            raise FileNotFoundError(f"Annotation file {json_filepath} does not exist.")
        
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