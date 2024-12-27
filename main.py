import cv2
import numpy as np
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Process video for corner tracking.')
parser.add_argument('input_video', type=str, help='Path to the input video file')
args = parser.parse_args()

# Remove extension from input video name to set output video name
input_filename = os.path.splitext(os.path.basename(args.input_video))[0]

# Load video
cap = cv2.VideoCapture(args.input_video)

# Select 4 corners with mouse clicks on first frame
def get_points(event, x, y, flags, param):
    global temp_frame, template_size
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        template_sizes.append(template_size)  # Save current template size
        cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_MOUSEMOVE and len(points) < 4:
        temp_frame = first_frame.copy()
        if len(points) > 0:
            for idx, (px, py) in enumerate(points):
                cv2.circle(temp_frame, (px, py), 5, (0, 0, 255), -1)
                top_left = (px - template_sizes[idx], py - template_sizes[idx])
                bottom_right = (px + template_sizes[idx], py + template_sizes[idx])
                cv2.rectangle(temp_frame, top_left, bottom_right, (255, 0, 0), 2)
        cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
        top_left = (x - template_size, y - template_size)
        bottom_right = (x + template_size, y + template_size)
        cv2.rectangle(temp_frame, top_left, bottom_right, (0, 255, 0), 2)

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Could not read video")
    cap.release()
    exit()

# Label corner points
points = []
template_sizes = []
template_size = 35
temp_frame = first_frame.copy()
cv2.namedWindow('Select Corners')
cv2.setMouseCallback('Select Corners', get_points)

while True:
    cv2.imshow('Select Corners', temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit
        break
    elif key == ord('2'):  # Press '2' to increase template size
        template_size += 5
        temp_frame = first_frame.copy()  # Update immediately after changing template size
        if len(points) > 0:
            for idx, (px, py) in enumerate(points):
                cv2.circle(temp_frame, (px, py), 5, (0, 0, 255), -1)
                top_left = (px - template_sizes[idx], py - template_sizes[idx])
                bottom_right = (px + template_sizes[idx], py + template_sizes[idx])
                cv2.rectangle(temp_frame, top_left, bottom_right, (255, 0, 0), 2)
    elif key == ord('1'):  # Press '1' to decrease template size
        template_size = max(5, template_size - 5)
        temp_frame = first_frame.copy()  # Update immediately after changing template size
        if len(points) > 0:
            for idx, (px, py) in enumerate(points):
                cv2.circle(temp_frame, (px, py), 5, (0, 0, 255), -1)
                top_left = (px - template_sizes[idx], py - template_sizes[idx])
                bottom_right = (px + template_sizes[idx], py + template_sizes[idx])
                cv2.rectangle(temp_frame, top_left, bottom_right, (255, 0, 0), 2)
    elif len(points) == 4:
        break

cv2.destroyAllWindows()

# Create template patches for each corner point
templates = []
for idx, (x, y) in enumerate(points):
    size = template_sizes[idx]
    template = first_frame[y-size:y+size, x-size:x+size]
    templates.append(template)

# Set up output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'{input_filename}_output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (first_frame.shape[1], first_frame.shape[0]))
out_perspective = cv2.VideoWriter(f'{input_filename}_output_perspective.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (first_frame.shape[1], first_frame.shape[0]))

# Process frames
prev_points = points
alpha = 1  # Low-pass filter coefficient
while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_points = []
    distances = []
    for i, (x, y) in enumerate(points):
        # Set search range
        search_margin = template_sizes[i] * 2
        top_left_x = max(0, x - search_margin)
        top_left_y = max(0, y - search_margin)
        bottom_right_x = min(frame.shape[1], x + search_margin)
        bottom_right_y = min(frame.shape[0], y + search_margin)

        search_area = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        res = cv2.matchTemplate(search_area, templates[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Calculate new corner position
        new_x = top_left_x + max_loc[0] + template_sizes[i]
        new_y = top_left_y + max_loc[1] + template_sizes[i]

        # Filter sudden movements (calculate distance from previous position)
        dist = np.sqrt((new_x - prev_points[i][0]) ** 2 + (new_y - prev_points[i][1]) ** 2)
        distances.append(dist)
        if dist > search_margin:  # Filter if movement exceeds threshold
            new_x, new_y = prev_points[i]

        new_points.append((new_x, new_y))

    # Calculate average velocity of other points
    avg_distance = np.mean([distances[j] for j in range(len(distances)) if j != i]) if len(distances) > 1 else 0
    adjusted_points = []

    for i, (new_x, new_y) in enumerate(new_points):
        # Calculate vector from previous position
        dx = new_x - prev_points[i][0]
        dy = new_y - prev_points[i][1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Limit velocity if faster than average
        if dist > avg_distance:
            scale = avg_distance / dist if dist != 0 else 0
            new_x = int(prev_points[i][0] + dx * scale)
            new_y = int(prev_points[i][1] + dy * scale)

        # Apply low-pass filter
        filtered_x = int(alpha * new_x + (1 - alpha) * prev_points[i][0])
        filtered_y = int(alpha * new_y + (1 - alpha) * prev_points[i][1])
        adjusted_points.append((filtered_x, filtered_y))

        # Draw point at new position
        cv2.circle(frame, (filtered_x, filtered_y), 5, (0, 255, 0), -1)

        # Visualize template
        top_left = (filtered_x - template_sizes[i], filtered_y - template_sizes[i])
        bottom_right = (filtered_x + template_sizes[i], filtered_y + template_sizes[i])
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    # Apply perspective transform
    if len(adjusted_points) == 4:
        width, height = first_frame.shape[1], first_frame.shape[0]
        pts1 = np.float32(adjusted_points)
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_frame = cv2.warpPerspective(frame, matrix, (width, height))
        out_perspective.write(perspective_frame)
        cv2.imshow('Perspective Frame', perspective_frame)

    # Save current frame to video
    out.write(frame)

    # Display frame
    cv2.imshow('Tracked Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

    # Update points
    prev_points = points
    points = adjusted_points

# Release all resources
cap.release()
out.release()
out_perspective.release()

# Wait 1 second before closing windows
cv2.waitKey(1000)
cv2.destroyAllWindows()
