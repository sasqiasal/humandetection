import cv2
import requests
import pygame

SERVER_URL = "http://10.208.91.175:8000/detect" 
pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")

cap = cv2.VideoCapture(0)

was_detected = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    _, img_encoded = cv2.imencode(".jpg", frame_resized)
    files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

    try:
        response = requests.post(SERVER_URL, files=files, timeout=2)
        result = response.json()

        person_detected = result.get("person_detected", False)
        detections = result.get("detections", [])

        for det in detections:
            label = det.get("label", "")
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            if label.lower() == "person":
                person_detected = True
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame_resized, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if person_detected and not was_detected:
            print("Orang terdeteksi! Alarm berbunyi!")
            pygame.mixer.music.stop()
            pygame.mixer.music.play()
            was_detected = True

    
        if not person_detected:
            was_detected = False

        if pygame.mixer.music.get_busy():
            cv2.putText(frame_resized, "ORANG TERDETEKSI", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    except Exception as e:
        print("Gagal mengirim ke server:", e)

    cv2.imshow("Client Camera", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()