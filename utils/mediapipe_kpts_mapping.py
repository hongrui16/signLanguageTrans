
import numpy as np


class mediapipeKptsMapping:
    def __init__(self):            
        '''
        hand keypoints indices (0–20)
        0: Wrist
        1: Thumb CMC
        2: Thumb MCP
        3: Thumb IP
        4: Thumb Tip
        5: Index Finger MCP
        6: Index Finger PIP
        7: Index Finger DIP
        8: Index Finger Tip
        9: Middle Finger MCP
        10: Middle Finger PIP
        11: Middle Finger DIP
        12: Middle Finger Tip
        13: Ring Finger MCP
        14: Ring Finger PIP
        15: Ring Finger DIP
        16: Ring Finger Tip
        17: Little Finger MCP
        18: Little Finger PIP
        19: Little Finger DIP
        20: Little Finger Tip
        '''

        self.hand_keypoints_mapping = {
            0: 0, 
            1: 1, 2: 2, 3: 3, 4: 4,  # Thumb
            5: 5, 6: 6, 7: 7, 8: 8,  # Index finger
            9: 9, 10: 10, 11: 11, 12: 12,  # Middle finger
            13: 13, 14: 14, 15: 15, 16: 16,  # Ring finger
            17: 17, 18: 18, 19: 19, 20: 20  # Little finger
            
        }
        n_hand_keypoints = len(self.hand_keypoints_mapping)

        self.hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),   # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),   # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]

        self.hand_adj_matrix = np.zeros((n_hand_keypoints, n_hand_keypoints), dtype=np.float32)  # 21 keypoints



        '''
        face Keypoint Indices (0–17)
        Face outline (0–5 for right and left eyebrows):
        mediapipe rightEyebrowLower: [46, 52, 55] order from right to left
        mediapipe leftEyebrowLower: [285, 282, 276 ],order from right to left


        Mouth (6–17):
        61, 40, 37, 0, 267, 270, 
        291, 321, 314, 17, 181, 91. 

        '''
        self.face_keypoints_mapping = {
            0: 46, 1: 52, 2: 55,  # Right eyebrow
            3: 285, 4: 282, 5: 276,  # Left eyebrow
            6: 61, # right mouth corner
            7: 40, 8: 37, 9: 0, 10: 267, 11: 270,  # Upper lip
            12: 291, # left mouth corner
            13: 321, 14: 314, 15: 17, 16: 181, 17: 91  # Lower lip
        }

        # Define the number of keypoints (18 for face)
        n_keypoints_faces = len(self.face_keypoints_mapping)

        # Initialize a zero matrix (18x18) on the appropriate device (e.g., CPU or GPU)
        self.face_adj_matrix = np.zeros((n_keypoints_faces, n_keypoints_faces), dtype=np.float32)  # Use torch.float32 for consistency

        # Define connections (edges) for 18 facial keypoints
        self.face_edges = [
            # Face outline
            (0, 1), (1, 2), # Right eyebrow
            (3, 4), (4, 5),  # left eyebrow
            
            # Mouth
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),  # Upper lip
            (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # Lower lip
            (16, 17)  
        ]


        '''
        body Keypoint Indices (0–8)
        0: Right wrist --> 16
        1: Right elbow --> 14
        2: Right shoulder --> 12
        3: Left wrist --> 15
        4: Left elbow --> 13
        5: Left shoulder --> 11
        6: Right eye --> 5
        7: Left eye --> 2
        8: Nose --> 0
        

        '''
        self.body_keypoints_mapping = {
            0: 16,  # right wrist
            1: 14,  # right elbow
            2: 12,  # Right shoulder
            3: 15,  # Left wrist
            4: 13,  # Left elbow
            5: 11,  # Left shoulder
            6: 5, # Right eye
            7: 2, # Left eye
            8: 0  # Nose

        }


        n_keypoints_body = len(self.body_keypoints_mapping)

        # Initialize a zero matrix (9x9) on the appropriate device (e.g., CPU or GPU)
        self.body_adj_matrix = np.zeros((n_keypoints_body, n_keypoints_body), dtype=np.float32)  # Use torch.float32 for consistency

        # Define connections (edges) for 9 body keypoints
        self.body_edges = [
            (0, 1), (1, 2),  # Right arm
            (3, 4), (4, 5),  # Left arm
            (2, 8), (5, 8),  # Shoulders to nose
            (6, 8), (7, 8)  # Eyes to nose            
        ]

        self._build_ajd_matrix()

    def _build_ajd_matrix(self):
        
        for i, j in self.hand_edges:
            self.hand_adj_matrix[i, j] = 1
            self.hand_adj_matrix[j, i] = 1
        
        for i, j in self.face_edges:
            self.face_adj_matrix[i, j] = 1
            self.face_adj_matrix[j, i] = 1

        for i, j in self.body_edges:
            self.body_adj_matrix[i, j] = 1
            self.body_adj_matrix[j, i] = 1

MediapipeKptsMapping = mediapipeKptsMapping()