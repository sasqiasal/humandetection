import cv2
import time
import multiprocessing as mp
import requests
import traceback

# ---- CONFIG ----
cam1_url = "http://192.168.1.16:8080/video"  # IP Webcam HP 1
cam2_url = "http://192.168.1.10:8080/video"  # IP Webcam HP 2
USE_YOLO = True
SHOW_FPS = True

# Raspberry Pi URLs
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

def worker(name, cam_url, run_yolo=False):
    was_detected = False
    try:
        cap = cv2.VideoCapture(cam_url)
        if not cap.isOpened():
            print(f"[{name}] Gagal buka stream: {cam_url}")
            return
        print(f"[{name}] Stream terbuka.")

        prev_time = time.time()
        frames = 0
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[{name}] Gagal ambil frame, reconnect...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(cam_url)
                continue

            person_detected = False
            # ---- YOLO ----
            if run_yolo and yolo_model is not None:
                try:
                    results = yolo_model(frame, conf=0.35, iou=0.45)
                    person_detected = False
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        if cls_id == 0:  # person
                            person_detected = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0].item()
                            label = f"Person {conf:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                            cv2.putText(frame, label, (x1, y1-10),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                except Exception as e:
                    print(f"[{name}] YOLO runtime error:", e)

            # ---- Kirim status ke Raspberry Pi ----
            try:
                resp = requests.get(RASPI_URL_STATUS, timeout=0.5)
                playing = resp.json().get("playing", False)
            except:
                playing = False

            # ---- Play alarm jika ada orang terdeteksi ----
            if person_detected and not was_detected and not playing:
                try:
                    requests.get(RASPI_URL_PLAY, timeout=0.5)
                    print(f"[{name}] Alarm diputar di Raspberry Pi")
                    playing = True
                except Exception as e:
                    print(f"[{name}] Gagal play alarm:", e)
                was_detected = True

            if not person_detected:
                was_detected = False

            # ---- FPS ----
            frames += 1
            if SHOW_FPS:
                now = time.time()
                if now - prev_time >= 1.0:
                    fps = frames / (now - prev_time)
                    prev_time = now
                    frames = 0
                cv2.putText(frame, f"{name} FPS: {fps:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # ---- Overlay saat musik diputar ----
            if playing:
                text = "ALARM SEDANG DIPUTAR!"
                cv2.putText(frame, text, (50, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

            # ---- Tampilkan frame ----
            cv2.imshow(name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(name)
        print(f"[{name}] Selesai.")

    except KeyboardInterrupt:
        print(f"[{name}] KeyboardInterrupt, keluar.")
    except Exception:
        print(f"[{name}] Error tak terduga:")
        traceback.print_exc()
    finally:
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

def main():
    print("[MAIN] Mulai multiproses HP Webcam...")
    processes = []
    p1 = mp.Process(target=worker, args=("HP-Cam-1", cam1_url, USE_YOLO), daemon=True)
    p2 = mp.Process(target=worker, args=("HP-Cam-2", cam2_url, USE_YOLO), daemon=True)
    processes.extend([p1, p2])

    for p in processes:
        p.start()
        time.sleep(0.5)

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt, hentikan semua worker...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
    finally:
        print("[MAIN] Semua proses dihentikan. Bye.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
