import cv2
from ultralytics import YOLO
import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")  # Ganti nama file sesuai punya kamu

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

previous_detected = False
alarm_ready = True
empty_frame_count = 0
empty_frame_threshold = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    detected = False

    for box in boxes:
        class_id = int(box.cls[0])
        if class_id == 0:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Seseorang", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Reset alarm
    if detected:
        if alarm_ready:
            pygame.mixer.music.play()
            alarm_ready = False 
        empty_frame_count = 0  
    else:
        empty_frame_count += 1
        if empty_frame_count >= empty_frame_threshold:
            alarm_ready = True  

    cv2.imshow("Deteksi Manusia", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
