from fastapi import FastAPI, File, UploadFile
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")  

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)[0]
    detected = False
    for cls in results.boxes.cls:
        if model.names[int(cls)] == "person":
            detected = True
            break

    return {"person_detected": detected}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
