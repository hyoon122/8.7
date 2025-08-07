import cv2
import dlib

# 모자이크 비율
MOSAIC_RATE = 15

# 얼굴 검출기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

