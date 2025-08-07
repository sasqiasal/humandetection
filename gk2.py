import cv2
import time
import pygame
from ultralytics import YOLO
import threading

# Inisialisasi YOLOv8
model = YOLO('yolov8n.pt')  # atau 'yolov8s.pt' jika ingin lebih akurat

# Hanya deteksi manusia
TARGET_CLASS = 'person'

# Inisialisasi pygame untuk suara
pygame.mixer.init()
sound = pygame.mixer.Sound("warning.mp3")  # Pastikan file ini ada

# IP Webcam Android, ganti sesuai IP HP kamu
ip_cam_url = 'http://192.168.1.30:8080/video'

# Buka kamera 0 (laptop) dan IP Webcam (HP)
cams = [
    cv2.VideoCapture(0),
    cv2.VideoCapture(ip_cam_url)
]

# Variabel kontrol suara
last_detect_time = 0
cooldown = 5  # detik
detected_recently = False

def play_sound_once():
    pygame.mixer.Sound.play(sound)

while True:
    person_detected = False
    for i, cam in enumerate(cams):
        ret, frame = cam.read()
        if not ret:
            print(f"Kamera {i} tidak terbaca.")
            continue

        # Jalankan deteksi
        results = model(frame, verbose=False)[0]

        # Cek apakah ada 'person'
        for r in results.boxes.cls:
            label = model.names[int(r)]
            if label == TARGET_CLASS:
                person_detected = True
                break

        # Tampilkan frame
        cv2.imshow(f"Kamera {i}", frame)

    current_time = time.time()

    if person_detected:
        if not detected_recently or current_time - last_detect_time > cooldown:
            print("🚨 Manusia terdeteksi! Bunyi alarm.")
            threading.Thread(target=play_sound_once).start()
            last_detect_time = current_time
            detected_recently = True
    else:
        if current_time - last_detect_time > cooldown:
            detected_recently = False

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan semua
for cam in cams:
    cam.release()
cv2.destroyAllWindows()
