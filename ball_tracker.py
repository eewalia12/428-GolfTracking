import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt

model = YOLO('best.pt')

def collect_frames(video_path):
    imgs = []
    cap = cv2.VideoCapture(video_path)
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            imgs.append((frame))
    return imgs

def process_video(video_path):
    frames = collect_frames(video_path)
    tracker = cv2.TrackerCSRT_create()
    tracking_initialized = False
    ball_positions = []

    for i, frame in enumerate(frames[:-1]):
        # Re-detect ball when tracking is lost or every 5 frames
        if not tracking_initialized or i % 5 == 0: 
            results = model.predict(source=frame, save=False, imgsz=640)
            detected_ball = None
            highest_confidence = 0

            for result in results:
                for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    if conf > highest_confidence:
                        highest_confidence = conf
                        best_ball = (x1, y1, x2 - x1, y2 - y1)

            if best_ball:
                detected_ball = best_ball

            if detected_ball:
                tracker.init(frame, detected_ball)
                tracking_initialized = True

        if tracking_initialized:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                ball_positions.append((x + w // 2, y + h // 2))

                # Bounding box and trajectory
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for j in range(1, len(ball_positions)):
                    cv2.line(frame, ball_positions[j - 1], ball_positions[j], (255, 0, 0), 5)

        # Display frame
        cv2.imshow('Golf Ball Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    # use path to either IMG_1636.mov or putt_test_2.mov
    video_path = "IMG_1636.mov"
    process_video(video_path)

if __name__ == "__main__":
    main()