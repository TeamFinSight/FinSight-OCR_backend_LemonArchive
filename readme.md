# 금융 서식 OCR API 서버

손글씨가 포함된 금융 서식 이미지에서 텍스트를 탐지하고 인식하는 FastAPI 기반의 OCR API 서버입니다.

## 주요 기능

- **이미지 기반 텍스트 추출**: 이미지 파일(jpg, png 등)을 업로드하여 포함된 텍스트를 추출합니다.
- **JSON 데이터 반환**: 탐지된 텍스트의 내용과 위치 좌표(Bounding Box)를 구조화된 JSON 형식으로 반환합니다.
- **시각화 결과 제공**: OCR 결과가 원본 이미지 위에 네모 박스와 텍스트로 표시된 시각화 이미지의 URL을 함께 반환하여 직관적인 확인이 가능합니다.

## 기술 스택

- **Backend**: Python 3.9, FastAPI
- **ML/CV**: PyTorch, OpenCV, Timm, Pillow
- **Models**:
  - Text Detection: DBNet (ResNet18 Backbone)
  - Text Recognition: CRNN (EfficientNet Backbone + BiLSTM + CTC)
- **Dependencies**: pyclipper, shapely

## API 엔드포인트

### 이미지 처리 및 OCR 실행

- **Endpoint**: `POST /ocr/process_image`
- **Description**: 이미지를 업로드하여 OCR 처리를 수행하고, 결과 데이터와 시각화 이미지 URL을 반환합니다.
- **Request Body**: `multipart/form-data`
  - `file`: 이미지 파일
- **Success Response**: `200 OK`
  ```json
  {
    "ocr_results": [
      {
        "box": [
          [150, 200],
          [450, 200],
          [450, 250],
          [150, 250]
        ],
        "text": "예금주명"
      }
    ],
    "visualized_image_url": "[http://127.0.0.1:8000/static/outputs/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.jpg](http://127.0.0.1:8000/static/outputs/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.jpg)"
  }
  ```

---

## 설치 및 실행 방법

### 1. 사전 요구사항

- Git
- Anaconda 또는 Miniconda

### 2. 프로젝트 클론

```bash
git clone <your_repository_url>
cd <your_project_directory>
```

### 3. Conda 가상 환경 생성 및 활성화

```bash
# Python 3.9 기반의 'ocr-env' 가상 환경 생성
conda create --name ocr-env python=3.9 -y

# 가상 환경 활성화
conda activate ocr-env
```

### 4. 라이브러리 설치

**중요:** PyTorch는 `requirements.txt`와 별도로 먼저 설치해야 합니다.

```bash
# 4-1. PyTorch CPU 버전 수동 설치
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 4-2. requirements.txt 파일로 나머지 라이브러리 설치
pip install -r requirements.txt
```

### 5. 모델 파일 준비

1.  프로젝트 루트에 `models` 폴더를 생성합니다.
2.  학습된 모델 가중치 파일들(탐지 모델, 인식 모델)과 문자 맵 파일을 `models` 폴더 안으로 복사합니다.
3.  `ocr_pipeline.py` 파일 하단의 `TODO` 부분을 실제 모델 파일 경로에 맞게 수정합니다.
    ```python
    # ocr_pipeline.py

    # TODO: 아래 경로들을 실제 파일 위치에 맞게 수정해주세요.
    DET_WEIGHTS_PATH = "models/YOUR_DETECTION_MODEL.pth"
    REC_WEIGHTS_PATH = "models/YOUR_RECOGNITION_MODEL.pth"
    CHAR_MAP_PATH = "models/korean_char_map.txt"
    ```

### 6. API 서버 실행

```bash
# uvicorn을 사용하여 FastAPI 개발 서버 실행
uvicorn main:app --reload
```
서버가 성공적으로 실행되면 터미널에 `Uvicorn running on http://127.0.0.1:8000` 메시지가 나타납니다.

### 7. API 테스트

1. 웹 브라우저를 열고 `http://127.0.0.1:8000/docs` 로 접속합니다.
2. `POST /ocr/process_image` 엔드포인트를 찾아 확장합니다.
3. `Try it out` -> `Choose File` 버튼을 눌러 테스트할 이미지 파일을 선택합니다.
4. `Execute` 버튼을 눌러 API를 실행하고 결과를 확인합니다.

## 프로젝트 구조

```
.
├── static/                # 결과 이미지 등 정적 파일 저장
│   └── outputs/
├── models/                # 모델 가중치(.pth), 문자 맵(.txt) 저장
├── main.py                # FastAPI 서버 메인 파일
├── ocr_pipeline.py        # OCR 모델 로딩 및 처리 로직
├── requirements.txt       # 프로젝트 의존성 라이브러리 목록
└── README.md              # 프로젝트 설명 파일
```