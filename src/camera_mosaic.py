import cv2
import dlib

# 모자이크 비율
MOSAIC_RATE = 15

# 얼굴 검출기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 가져올 수 없습니다.")
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)

    for rect in faces:
        x, y = rect.left(), rect.top()
        w, h = rect.right() - x, rect.bottom() - y

        # 얼굴 영역을 클리핑하고 모자이크 적용
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:  # 잘못된 영역 방지
            continue
        small = cv2.resize(roi, (w // MOSAIC_RATE, h // MOSAIC_RATE))
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
        frame[y:y+h, x:x+w] = mosaic

        # 얼굴 랜드마크 검출 (선택사항: 시각화용)
        shape = predictor(gray, rect)
        for i in range(68):
            part = shape.part(i)
            cv2.circle(frame, (part.x, part.y), 1, (0, 0, 255), -1)

    # 결과 출력
    cv2.imshow("Face Mosaic", frame)
    if cv2.waitKey(1) == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()
