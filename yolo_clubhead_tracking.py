## Track clubhead given a video of a golf swing
import cv2
import numpy as np
import math
import skimage
from roboflow import Roboflow

from keys import api_key

def define_yolo_model():
    rf = Roboflow(api_key=api_key())
    project = rf.workspace().project("golf-club-head")
    model = project.version(2).model

    return model

def detect_clubhead(yolo, frame):
    result = yolo.predict(frame, confidence=65, overlap=30).json()
    preds = result["predictions"]
    if preds:
        x = preds[0]['x']
        y = preds[0]['y']
        w = preds[0]['width']
        h = preds[0]['height']
        return int(x), int(y), int(w), int(h)
    return None, None, None, None

def collect_frames(video_path):
    imgs = []
    cap = cv2.VideoCapture(video_path)
    ret = True
    while ret:
        ret, image = cap.read()
        if ret:
            image = cv2.convertScaleAbs(image, 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(image, (3, 3), 0)
            imgs.append(image)
    return np.stack(imgs, axis=0)

def collect_clubhead_points(edges, center):
    clubhead_points = []
    for edge in edges:
        lines = skimage.transform.probabilistic_hough_line(edge, threshold=55, line_length=55, line_gap=4)
        max_dist, furthest_point = 0, None
        for line in lines:
            for point in line:
                if math.dist(point, center) > max_dist:
                    max_dist = math.dist(point, center)
                    furthest_point = point
        clubhead_points.append(furthest_point)
    return clubhead_points

def initialize_tracker(frame, bbox):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    return tracker

def image_difference(images):
    threshold = 7 # color distance threshold

    binary_differences, edges = [], []
    for i in range(len(images)-1):
        frame1 = images[i]
        frame2 = images[i+1]

        frame_diff = cv2.absdiff(frame1, frame2)
        _, binary_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        binary_diff = skimage.morphology.binary_closing(binary_diff)
        edge = skimage.feature.canny(binary_diff, sigma=1)
        edges.append(edge)
        binary_differences.append(binary_diff)

    binary_differences = np.array(binary_differences)
    edges = np.array(edges)
    return binary_differences, edges

def csrt_unmoving(roi_list):
    if len(roi_list) > 5:
        if math.dist(roi_list[-5], roi_list[-1]) < 5:
            return True
    return False

def main():
    # good tests: 1146, 754, 617, 591, 364, 552, 219
    yolo = define_yolo_model()

    video_path = "woods.mp4"
    frames = collect_frames(video_path)
    
    masks, edges = image_difference(frames)

    height, width = frames[0].shape
    center = (width // 2, height // 2)

    points = collect_clubhead_points(edges, center)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    tracker = None
    roi = None
    frame_id = 0
    seen_roi = []
    prev_locations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of Video")
            break

        if roi:
            seen_roi.append((roi[0], roi[1]))

        if not tracker:
            yolo_x, yolo_y, yolo_w, yolo_h = detect_clubhead(yolo, frame)
            if yolo_x and yolo_y:  # Use yolo detection for initial tracking
                roi = (yolo_x - (yolo_w // 2), yolo_y - (yolo_h // 2), yolo_w, yolo_h)
                tracker = initialize_tracker(frame, roi)
            elif frame_id < len(points) and points[frame_id] is not None:
                x, y = points[frame_id]  # Fallback to line detection if yolo fails
                roi = (x - 5, y - 5, 20, 10)
                tracker = initialize_tracker(frame, roi)

        # If tracking is still active, check for unmoving object
        if tracker:
            success, roi = tracker.update(frame)
            if success and not csrt_unmoving(seen_roi):
                x, y, w, h = map(int, roi)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                prev_locations.append((x, y))
            else: 
                yolo_x, yolo_y, yolo_w, yolo_h = detect_clubhead(yolo, frame)
                if yolo_x:
                    roi = (yolo_x - (yolo_w // 2), yolo_y - (yolo_h // 2), yolo_w, yolo_h)
                    tracker = initialize_tracker(frame, roi)
                elif frame_id < len(points) and points[frame_id] is not None:
                    x, y = points[frame_id]
                    roi = (x - 5, y - 5, 20, 10)
                    tracker = initialize_tracker(frame, roi)

        if len(prev_locations) > 1:
            for i in range(1, len(prev_locations)):
                cv2.line(frame, prev_locations[i-1], prev_locations[i], (15, 165, 145),thickness=3)

        cv2.imshow("Golf Clubhead Tracking", frame)
        frame_id += 1

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
