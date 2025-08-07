import cv2
from ultralytics import YOLO
import pygame

# Inisialisasi suara
pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")

# Load model YOLO
model = YOLO("yolov8n.pt")

# List kamera: 0 = laptop, 1 = IP kamera (ubah sesuai IP Webcam)
cam_urls = [0, "http://192.168.1.30:8080/video"]  # ganti IP sesuai IP Webcam HP-mu

caps = []
for url in cam_urls:
    cap = cv2.VideoCapture(url)
    caps.append(cap)

# Untuk kontrol alarm
alarm_ready = True
empty_frame_count = 0
empty_frame_threshold = 5

while True:
    detected = False
    frames = []

    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            continue

        results = model(frame)
        boxes = results[0].boxes

        for box in boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Deteksi hanya manusia
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Manusia", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frames.append(frame)

    if detected:
        if alarm_ready:
            pygame.mixer.music.play()
            alarm_ready = False
        empty_frame_count = 0
    else:
        empty_frame_count += 1
        if empty_frame_count >= empty_frame_threshold:
            alarm_ready = True

    # Tampilkan semua frame
    for i, frame in enumerate(frames):
        if frame is not None:
            cv2.imshow(f"Kamera {i+1}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release semua kamera
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
pygame.mixer.quit()
