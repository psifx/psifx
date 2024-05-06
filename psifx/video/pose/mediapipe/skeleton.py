"""Mediapipe's skeleton graph.'"""

from mediapipe.python.solutions.holistic import (
    PoseLandmark,
    HandLandmark,
    POSE_CONNECTIONS,
    HAND_CONNECTIONS,
    FACEMESH_TESSELATION,
)
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS


N_POSE_LANDMARKS = len([p.value for p in PoseLandmark])
N_FACE_LANDMARKS = FACEMESH_NUM_LANDMARKS
N_LEFT_HAND_LANDMARKS = len([p.value for p in HandLandmark])
N_RIGHT_HAND_LANDMARKS = len([p.value for p in HandLandmark])

POSE_EDGES = tuple(POSE_CONNECTIONS)
FACE_EDGES = tuple(FACEMESH_TESSELATION)
# FACE_EDGES = tuple(FACEMESH_CONTOURS)
# FACE_EDGES = tuple(FACEMESH_TESSELATION) + tuple(FACEMESH_CONTOURS)
LEFT_HAND_EDGES = tuple(HAND_CONNECTIONS)
RIGHT_HAND_EDGES = tuple(HAND_CONNECTIONS)
