import cv2
import dlib
from scipy.spatial import distance

# EAR 계산 함수
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 눈 좌표 인덱스 (dlib의 68개 랜드마크 기준)
LEFT_EYE_INDEXES = list(range(36, 42))
RIGHT_EYE_INDEXES = list(range(42, 48))

# EAR 임계값 및 프레임 지속 조건
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

