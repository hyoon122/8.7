import cv2
import dlib
from scipy.spatial import distance
import time

print("If you want to cancel this, press 'ESC' button")

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

# EAR 임계값 (프레임에서 시간'초'로 변경.)
EAR_THRESHOLD = 0.25
EYE_CLOSED_DURATION_THRESHOLD = 3  # 3초

# dlib 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 상태 변수
eye_closed_start_time = None

# 졸음 프레임 카운터
sleep_frame_count = 0

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # 왼쪽 눈, 오른쪽 눈 좌표 추출
        left_eye = [landmarks[i] for i in LEFT_EYE_INDEXES]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDEXES]

        # EAR 계산
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # 눈 닫힘 감지
        if avg_ear < EAR_THRESHOLD:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            else:
                elapsed = time.time() - eye_closed_start_time
                if elapsed >= EYE_CLOSED_DURATION_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            eye_closed_start_time = None  # 눈 떴으면 초기화

        # 눈 윤곽선 그리기 (옵션)
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # 결과 출력
    cv2.imshow("Drowsiness Detection", frame)

    # 종료 조건 (ESC)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
