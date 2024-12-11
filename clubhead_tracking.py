## Track clubhead given a video of a golf swing
import cv2
import numpy as np
import math
from collections import defaultdict
from matplotlib import pyplot as plt
import skimage
from skimage.filters import threshold_otsu

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

def show_frames(frames, masks, edges, lines):
    plt.ion()  # Turn on interactive mode
    all_points = []
    for i, mask in enumerate(masks):
        plt.clf()  # Clear the figure for the new frame
        # Show the original frame
        plt.subplot(2, 3, 1)
        plt.imshow(frames[i], cmap='gray')
        plt.title('Frame')
        plt.axis('off')

        # Show the mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        # Show the edges
        plt.subplot(2, 3, 3)
        if edges[i] is not None:
            plt.imshow(edges[i], cmap='gray')
        plt.title('Edges')
        plt.axis('off')

                # Show the lines overlaid on the frame
        plt.subplot(2, 3, 4)
        frame_with_lines = np.copy(frames[i])
        if lines[i]:
            for line in lines[i]:
                p0, p1 = line
                cv2.line(frame_with_lines, (p0[0], p0[1]), (p1[0], p1[1]), (255,), 2)
                cv2.circle(frame_with_lines, (p0[0], p0[1]), radius=5, color=(255,), thickness=-1)
                # cv2.circle(frame_with_lines, (p1[0], p1[1]), radius=5, color=(255,), thickness=-1) 
        plt.imshow(frame_with_lines, cmap='gray')
        plt.title('Lines')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        frame_with_points = np.copy(frames[i])
        if lines[i]:
            for line in lines[i]:
                p0, p1 = line
                all_points.append(p0)
                all_points.append(p1)

        for point in all_points:
            cv2.circle(frame_with_points, (point[0], point[1]), radius=10, color=(255,), thickness=-1)
        plt.imshow(frame_with_points, cmap='gray')
        plt.title('Lines')
        plt.axis('off')

        plt.pause(0.01)
        plt.draw()

def collect_lines(edges):
    hough_lines = []
    for edge in edges:
        lines = skimage.transform.probabilistic_hough_line(edge, threshold=55, line_length=55, line_gap=4)
        max_line_dist, max_line = 0, None
        for line in lines:
            if math.dist(line[0], line[1]) > max_line_dist:
                max_line_dist = math.dist(line[0], line[1])
                max_line = [line]
        hough_lines.append(max_line)
    return hough_lines

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
    video_path = "woods.mp4"
    frames = collect_frames(video_path)
    
    masks, edges = image_difference(frames)
    # lines = collect_lines(edges)
    # show_frames(frames, masks, edges, lines)
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
            if frame_id < len(points) and points[frame_id] is not None:
                x, y = points[frame_id]
                roi = (x-5, y-5, 20, 10)
                tracker = initialize_tracker(frame, roi)

        if frame_id < len(points) and points[frame_id]:
            if (tracker and math.dist(center, points[frame_id]) > (math.dist(center, (roi[0], roi[1])) + 40)) or csrt_unmoving(seen_roi):  
                roi = (points[frame_id][0]-5, points[frame_id][1]-5, 20, 10)
                tracker = initialize_tracker(frame, roi)

        if tracker:
            success, roi = tracker.update(frame)
            if success:
                x, y, w, h = map(int, roi)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                prev_locations.append((x, y))
            else:
                tracker = None

        if len(prev_locations) > 1:
            for i in range(1, len(prev_locations)):
                cv2.line(frame, prev_locations[i-1], prev_locations[i], (15, 165, 145),thickness=3)

        cv2.imshow("CSRT Object Tracking", frame)
        frame_id += 1

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()