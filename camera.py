import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Use yolo11s since it is lightweight and better for realtime
model = YOLO('yolo11s.pt') 

def detect_from_camera():
    """
    Runs real time YOLO object detection on video from the webcam
    """
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam, use another if not working
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Run YOLO object detection on the current frame
        results = model(frame)

        result_img = results[0].plot()  # Visualize on the frame

        cv2.imshow('YOLO Object Detection', result_img)

        # Exit if Q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Launching...")
    detect_from_camera()
