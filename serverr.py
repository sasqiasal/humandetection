from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from threading import Lock

app = FastAPI()
model = YOLO("yolov8n.pt")  

# Untuk simpan frame terakhir dari client
latest_frame = None
frame_lock = Lock()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_data = await file.read()
    npimg = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.5)
    detections = []
    person_detected = False

    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        label = model.names[cls_id]
        if label == "person":
            person_detected = True
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append({
                "label": label,
                "bbox": [x1, y1, x2, y2]
            })

    return JSONResponse(content={
        "person_detected": person_detected,  # ✅ boolean, bukan list
        "detections": detections
    })

@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    """Terima frame dari client untuk ditampilkan di /video_feed"""
    global latest_frame
    image_data = await file.read()
    npimg = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    with frame_lock:
        latest_frame = frame

    return {"status": "frame_received"}

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
