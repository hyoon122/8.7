import cv2
import numpy as np
import os, glob


# 변수 설정
base_dir = './faces'
train_data, train_labels = [], []

print("📍 현재 작업 디렉토리:", os.getcwd())
print("📁 base_dir 절대경로:", os.path.abspath(base_dir))

# 폴더 리스트 수집
dirs = [d for d in glob.glob(base_dir + "/*") if os.path.isdir(d)]
print('Collecting train data set:')

for folder_path in dirs:
    folder_name = os.path.basename(folder_path)   # 예: hy_0122
    try:
        id = folder_name.split('_')[1]            # 예: 0122
    except IndexError:
        print(f"⚠️ 폴더 이름 형식 오류: {folder_path}")
        continue

    # ✅ 디버깅용 출력 시작
    print(f"📂 검사 중인 폴더: {folder_path}")
    print(f"📁 절대경로: {os.path.abspath(folder_path)}")

    image_files = glob.glob(folder_path + '/*.jpg')
    print(f"🔍 찾은 이미지 파일들: {image_files}")
    # ✅ 디버깅용 출력 끝

    for file in image_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ 이미지 읽기 실패: {file}")
            continue
        print(f"✅ 이미지 읽음: {file}, shape: {img.shape}")
        if img.shape != (200, 200):
            print(f"⚠️ 이미지 크기 이상함: {file}, shape: {img.shape}")
            img = cv2.resize(img, (200, 200))  # 자동 리사이즈
        train_data.append(np.asarray(img, dtype=np.uint8))
        train_labels.append(int(id))

# 이미지 수 확인
print(f"✅ 수집된 이미지 개수: {len(train_data)}")
print(f"✅ 수집된 라벨 개수: {len(train_labels)}")

# NumPy 배열로 변환
train_data = np.asarray(train_data)
train_labels = np.int32(train_labels)

# 데이터가 충분한지 확인
if len(train_data) < 2:
    print("❗ Error: 학습 데이터가 부족합니다. 최소 2장 이상의 얼굴 이미지가 필요합니다.")
    exit()

# LBP 얼굴 인식기 생성 및 훈련
print('Starting LBP Model training...')
model = cv2.face.LBPHFaceRecognizer_create()
model.train(train_data, train_labels)
model.write('../faces/all_face.xml')
print("✅ Model trained successfully!")
