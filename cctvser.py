import cv2
import time
import multiprocessing as mp
import requests
import traceback
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

# ---- CONFIG ----
cam1_url = "http://192.168.1.16:8080/video"
cam2_url = "http://192.168.1.10:8080/video"
USE_YOLO = True
FRAME_WIDTH = 480  # resize untuk dashboard
FRAME_HEIGHT = 360

RASPI_URL_PLAY = "http://192.168.1.20:5000/play_sound"
RASPI_URL_STATUS = "http://192.168.1.20:5000/sound_status"

# ---- Load YOLO ----
yolo_model = None
if USE_YOLO:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("[INFO] YOLO siap.")
    except Exception as e:
        print("[WARN] YOLO gagal load:", e)
        USE_YOLO = False

# ---- FastAPI ----
app = FastAPI()

# Worker function
def worker(name, cam_url, frames_dict):
    was_detected = False
    try:
        cap = cv2.VideoCapture(cam_url)
        if not cap.isOpened():
            print(f"[{name}] Gagal buka stream: {cam_url}")
            return
        print(f"[{name}] Stream terbuka.")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(cam_url)
                continue

            person_detected = False
            if USE_YOLO and yolo_model:
                try:
                    results = yolo_model(frame, conf=0.35, iou=0.45)
                    frame = results[0].plot()
                    if len(results[0].boxes) > 0:
                        person_detected = True
                except Exception as e:
                    print(f"[{name}] YOLO runtime error:", e)

            # Cek status musik
            try:
                resp = requests.get(RASPI_URL_STATUS, timeout=0.5)
                playing = resp.json().get("playing", False)
            except:
                playing = False

            # Play alarm jika ada orang
            if person_detected and not was_detected and not playing:
                try:
                    requests.get(RASPI_URL_PLAY, timeout=0.5)
                    playing = True
                    print(f"[{name}] Alarm diputar di Raspberry Pi")
                except Exception as e:
                    print(f"[{name}] Gagal play alarm:", e)
                was_detected = True
            if not person_detected:
                was_detected = False

            # Overlay teks saat musik diputar
            if playing:
                cv2.putText(frame, "ALARM SEDANG DIPUTAR!", (50, frame.shape[0]-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            # Resize frame untuk dashboard
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Update shared dict
            frames_dict[name] = frame_resized

    except Exception:
        print(f"[{name}] Error tak terduga:")
        traceback.print_exc()
    finally:
        try: cap.release()
        except: pass

# Generator untuk streaming
def generate_frames(cam_name, frames_dict):
    while True:
        frame = frames_dict.get(cam_name, None)
        if frame is None:
            time.sleep(0.01)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# HTML dashboard
@app.get("/")
def index():
    html_content = f"""
    <html>
        <head>
            <title>Dashboard CCTV HP Webcam</title>
            <style>
                body {{ background: #222; color: #fff; text-align: center; font-family: Arial; }}
                .camera {{ display: inline-block; margin: 10px; }}
                img {{ border: 3px solid #fff; }}
            </style>
        </head>
        <body>
            <h1>Dashboard CCTV HP Webcam</h1>
            <div class="camera">
                <h2>HP-Cam-1</h2>
                <img src="/video_feed/HP-Cam-1" width="{FRAME_WIDTH}" height="{FRAME_HEIGHT}">
            </div>
            <div class="camera">
                <h2>HP-Cam-2</h2>
                <img src="/video_feed/HP-Cam-2" width="{FRAME_WIDTH}" height="{FRAME_HEIGHT}">
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed/{cam_name}")
def video_feed(cam_name: str):
    if cam_name not in frames_dict:
        return {"error": "Camera not found"}
    return StreamingResponse(generate_frames(cam_name, frames_dict),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ---- Main ----
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Shared dict untuk frames
    manager = mp.Manager()
    frames_dict = manager.dict()
    frames_dict["HP-Cam-1"] = None
    frames_dict["HP-Cam-2"] = None

    # Start worker
    processes = [
        mp.Process(target=worker, args=("HP-Cam-1", cam1_url, frames_dict), daemon=True),
        mp.Process(target=worker, args=("HP-Cam-2", cam2_url, frames_dict), daemon=True)
    ]
    for p in processes:
        p.start()

    # Run FastAPI
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
