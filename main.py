from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles # 추가
from typing import Dict, List, Any
import ocr_pipeline
import cv2
import os # 추가
import uuid # 추가

app = FastAPI(title="OCR Project API")

# --- Static 파일 서빙 설정 (새로 추가된 부분) ---
# 'static'이라는 폴더를 만들고, /static 경로로 파일을 제공할 수 있도록 설정합니다.
os.makedirs("static/outputs", exist_ok=True) # 폴더가 없으면 생성
app.mount("/static", StaticFiles(directory="static"), name="static")
# ---------------------------------------------------


@app.get("/")
def read_root():
    return {"message": "OCR API 서버에 오신 것을 환영합니다! 모델이 로드되었습니다."}

@app.post("/ocr/process_image", response_model=Dict[str, Any])
async def process_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    ocr_results, visualized_image = ocr_pipeline.run_ocr(image_bytes)

    # --- Base64 인코딩 대신 파일로 저장하고 URL 생성 ---
    # 1. 고유한 파일 이름 생성 (덮어쓰기 방지)
    filename = f"{uuid.uuid4()}.jpg"
    save_path = os.path.join("static/outputs", filename)

    # 2. 시각화된 이미지를 파일로 저장
    cv2.imwrite(save_path, visualized_image)

    # 3. 클라이언트가 접근할 수 있는 URL 생성
    #    (실제 서버에서는 request.base_url을 사용하는 것이 더 좋습니다)
    image_url = f"http://127.0.0.1:8000/static/outputs/{filename}"
    # ----------------------------------------------------

    # 4. 최종 결과를 새로운 JSON 형태로 반환
    return {
        "ocr_results": ocr_results,
        "visualized_image_url": image_url
    }