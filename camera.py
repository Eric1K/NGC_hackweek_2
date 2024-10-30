import cv2
from ultralytics import YOLO
import time
from fastapi import Request

model = YOLO('yolo11s.pt')

async def detect_from_camera(request: Request):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            # Check if the client has disconnected
            if await request.is_disconnected():
                print("Client disconnected, releasing camera.")
                break

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                cap.release()
                cap = cv2.VideoCapture(0)
                time.sleep(1)
                continue

            results = model(frame)
            result_img = results[0].plot(show=False)

            ret, buffer = cv2.imencode('.jpg', result_img)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        cap.release()  # Ensure the camera is released once the loop is exited