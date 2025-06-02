import numpy as np

class openposeKptsMapping:
    def __init__(self):
        '''
        Hand Keypoints Indices (0–20)
        Same anatomical positions as MediaPipe:
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
            0: 0,  # Wrist
            1: 1, 2: 2, 3: 3, 4: 4,  # Thumb
            5: 5, 6: 6, 7: 7, 8: 8,  # Index finger
            9: 9, 10: 10, 11: 11, 12: 12,  # Middle finger
            13: 13, 14: 14, 15: 15, 16: 16,  # Ring finger
            17: 17, 18: 18, 19: 19, 20: 20  # Little finger
        }
        n_hand_keypoints = len(self.hand_keypoints_mapping)

        self.hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Little finger
        ]

        self.hand_adj_matrix = np.zeros((n_hand_keypoints, n_hand_keypoints), dtype=np.float32)

        '''
        Face Keypoints Indices (0–17)
        Same anatomical positions as MediaPipe:
        0–2: Right eyebrow (outer, middle, inner)
        3–5: Left eyebrow (outer, middle, inner)
        6: Right mouth corner
        7–11: Upper lip (right to left)
        12: Left mouth corner
        13–17: Lower lip (left to right)
        '''
        self.face_keypoints_mapping = {
            0: 17, 1: 19, 2: 21,  # Right eyebrow (outer, middle, inner)
            3: 22, 4: 24, 5: 26,  # Left eyebrow (outer, middle, inner)
            6: 48,  # Right mouth corner
            7: 49, 8: 50, 9: 51, 10: 52, 11: 53,  # Upper lip
            12: 54,  # Left mouth corner
            13: 55, 14: 56, 15: 57, 16: 58, 17: 59  # Lower lip
        }
        n_face_keypoints = len(self.face_keypoints_mapping)

        self.face_edges = [
            (0, 1), (1, 2),  # Right eyebrow
            (3, 4), (4, 5),  # Left eyebrow
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),  # Upper lip
            (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # Lower lip
            (16, 17),
            (17, 6)
        ]

        self.face_adj_matrix = np.zeros((n_face_keypoints, n_face_keypoints), dtype=np.float32)

        '''
        Body Keypoints Indices (0–8)
        Same anatomical positions as MediaPipe:
        0: Right wrist
        1: Right elbow
        2: Right shoulder
        3: Left wrist
        4: Left elbow
        5: Left shoulder
        6: Right eye
        7: Left eye
        8: Nose
        '''
        self.body_keypoints_mapping = {
            0: 4,  # Right wrist
            1: 3,  # Right elbow
            2: 2,  # Right shoulder
            3: 7,  # Left wrist
            4: 6,  # Left elbow
            5: 5,  # Left shoulder
            6: 15,  # Right eye
            7: 16,  # Left eye
            8: 0   # Nose
        }
        n_body_keypoints = len(self.body_keypoints_mapping)

        self.body_edges = [
            (0, 1), (1, 2),  # Right arm
            (3, 4), (4, 5),  # Left arm
            (2, 8), (5, 8),  # Shoulders to nose
            (6, 8), (7, 8)  # Eyes to nose
        ]

        self.body_adj_matrix = np.zeros((n_body_keypoints, n_body_keypoints), dtype=np.float32)

        # Build adjacency matrices
        self._build_adj_matrix()

    def _build_adj_matrix(self):
        for i, j in self.hand_edges:
            self.hand_adj_matrix[i, j] = 1
            self.hand_adj_matrix[j, i] = 1
        for i, j in self.face_edges:
            self.face_adj_matrix[i, j] = 1
            self.face_adj_matrix[j, i] = 1
        for i, j in self.body_edges:
            self.body_adj_matrix[i, j] = 1
            self.body_adj_matrix[j, i] = 1

# Instantiate the class
OpenposeKptsMapping = openposeKptsMapping()