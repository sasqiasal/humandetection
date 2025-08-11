import cv2
import requests
import pygame
import time

SERVER_IP = "192.168.1.9"  # ganti sesuai IP server FastAPI
DETECT_URL = f"http://{SERVER_IP}:8000/detect"
UPLOAD_URL = f"http://{SERVER_IP}:8000/upload_frame"

pygame.mixer.init()
pygame.mixer.music.load("voicehuman.mp3")

cap = cv2.VideoCapture(0)
prev_time = time.time()
fps = 0
was_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Resize biar ringan
    frame_resized = cv2.resize(frame, (640, 480))
    _, img_encoded = cv2.imencode(".jpg", frame_resized)
    files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

    try:
        # === Kirim ke server untuk deteksi ===
        response = requests.post(DETECT_URL, files=files, timeout=5)
        result = response.json()

        person_detected = result.get("person_detected", False)
        detections = result.get("detections", [])

        # Gambar kotak di frame_resized
        for det in detections:
            label = det.get("label", "")
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            if label.lower() == "person":
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame_resized, "Ada Orang", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Logika alarm + tulisan "Suara diputar"
        if person_detected and not was_detected:
            pygame.mixer.music.stop()
            pygame.mixer.music.play()
            was_detected = True

        if not person_detected:
            was_detected = False

        if pygame.mixer.music.get_busy():
            cv2.putText(frame_resized, "Suara Pengingat Diputar", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # === Kirim frame yang sudah ada kotak + teks ke server ===
        _, processed_img_encoded = cv2.imencode(".jpg", frame_resized)
        upload_files = {"file": ("processed.jpg", processed_img_encoded.tobytes(), "image/jpeg")}
        requests.post(UPLOAD_URL, files=upload_files, timeout=5)

    except Exception as e:
        print("Gagal mengirim ke server:", e)

    # Tampilkan di client
    cv2.putText(frame_resized, f"FPS: {fps:.1f}", (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 51, 0), 2)
    cv2.imshow("Client Camera", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
