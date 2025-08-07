# 얼굴 인식 어플리케이션
# 수집 -> 훈련 -> 인식 순으로 진행
import cv2
import numpy as np
import os, glob

# 기본 경로 및 변수 설정
base_dir = './faces'  # 얼굴 사진과 모델을 저장할 기본 폴더
face_classifier = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')  # 얼굴 검출기
target_cnt = 400      # 얼굴 데이터 수집 시 찍을 사진 수 목표
min_accuracy = 85     # 얼굴 인식 시 인정할 최소 정확도(%) 

# 1. 얼굴 사진 데이터 수집 함수
def collect_data():
    # 사용자 이름과 ID 입력받기
    name = input("Insert User Name(Only Alphabet): ")
    id = input("Insert User Id(Non-Duplicate number): ")
    save_dir = os.path.join(base_dir, f"{name}_{id}")  # 저장할 폴더명: name_id 형식

    # 폴더가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cnt = 0  # 저장한 사진 수 초기화
    cap = cv2.VideoCapture(0)  # 웹캠 열기

    while cap.isOpened():
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            print("Camera read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환 (얼굴 검출에 필요)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출

        if len(faces) == 1:
            (x,y,w,h) = faces[0]  # 얼굴 위치 정보
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)  # 얼굴 위치 사각형 표시

            face = gray[y:y+h, x:x+w]  # 얼굴 영역만 자르기
            face = cv2.resize(face, (200, 200))  # 크기 통일

            file_path = os.path.join(save_dir, f"{cnt}.jpg")  # 저장할 파일 경로
            cv2.imwrite(file_path, face)  # 얼굴 이미지 저장

            cv2.putText(frame, str(cnt), (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)  # 저장 개수 표시
            cnt += 1

        else:
            # 얼굴이 없거나 여러 개일 때 경고 메시지 출력
            msg = "no face." if len(faces) == 0 else "too many faces."
            cv2.putText(frame, msg, (10,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)

        cv2.imshow('Collecting Face Data', frame)  # 결과 프레임 보여주기

        # ESC 키 누르거나 목표 사진 수 달성 시 종료
        if cv2.waitKey(1) == 27 or cnt >= target_cnt:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collecting Samples Completed. Collected: {cnt}")

# 2. 얼굴 인식 모델 훈련 함수
def train_model():
    train_data, train_labels = [], []

    # base_dir 내 하위 폴더(사용자 별 폴더) 리스트 가져오기
    dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]

    print('Collecting train data set:')
    for dir in dirs:
        folder_name = os.path.basename(dir)
        try:
            # 폴더명에서 ID 분리 (예: "name_12" -> 12)
            id = folder_name.split('_')[1]
        except IndexError:
            print(f"폴더 이름 형식 오류: {dir}")
            continue

        # 해당 폴더 내 jpg 파일 리스트
        files = glob.glob(dir + '/*.jpg')
        print(f'\t path: {dir}, {len(files)} files')

        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # 이미지가 없거나 크기가 맞지 않으면 건너뜀
            if img is None or img.shape != (200, 200):
                continue
            train_data.append(np.asarray(img, dtype=np.uint8))
            train_labels.append(int(id))

    # 데이터가 2장 미만이면 학습 불가
    if len(train_data) < 2:
        print("학습 데이터가 부족합니다. 최소 2장 이상의 얼굴 이미지가 필요합니다.")
        return

    # numpy 배열로 변환
    train_data = np.asarray(train_data)
    train_labels = np.int32(train_labels)

    print('Starting LBP Model training...')
    model = cv2.face.LBPHFaceRecognizer_create()  # LBPH 얼굴 인식기 생성
    model.train(train_data, train_labels)         # 모델 훈련

    # 저장 폴더가 없으면 생성
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 훈련된 모델 저장
    model.save(os.path.join(base_dir, 'all_face.xml'))
    print("Model trained successfully!")

# 3. 얼굴 인식 함수 (웹캠 실시간 인식)
def recognize_face():
    model_path = os.path.join(base_dir, 'all_face.xml')

    # 모델 파일 존재 여부 체크
    if not os.path.exists(model_path):
        print("학습된 모델 파일이 없습니다. 먼저 학습을 진행하세요.")
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_path)  # 모델 불러오기

    # 사용자 이름과 ID 매핑 생성
    dirs = [d for d in glob.glob(base_dir + "/*") if os.path.isdir(d)]
    names = {}
    for dir in dirs:
        folder = os.path.basename(dir)
        try:
            name, id = folder.split('_')
            names[int(id)] = name
        except:
            pass

    cap = cv2.VideoCapture(0)  # 웹캠 시작

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)  # 얼굴 영역 사각형

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            # 얼굴 인식 예측 (label, confidence)
            label, confidence = model.predict(face)

            if confidence < 400:
                # 거리를 % 정확도로 변환 (작을수록 가까움)
                accuracy = int(100 * (1 - confidence/400))
                if accuracy >= min_accuracy and label in names:
                    msg = f'{names[label]} ({accuracy}%)'
                else:
                    msg = 'Unknown'
            else:
                msg = 'Unknown'

            # 텍스트 배경 및 출력
            txt_size, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
            cv2.rectangle(frame, (x, y - base - txt_size[1]), (x + txt_size[0], y), (0,255,255), -1)
            cv2.putText(frame, msg, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)

        # ESC 키 누르면 종료
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 모드 선택 안내문
    print("Select mode:")
    print("1: Collect face data")
    print("2: Train model")
    print("3: Recognize face")
    mode = input("Enter mode number (1/2/3): ").strip()

    if mode == '1':
        collect_data()
    elif mode == '2':
        train_model()
    elif mode == '3':
        recognize_face()
    else:
        print("Invalid mode selected.")
