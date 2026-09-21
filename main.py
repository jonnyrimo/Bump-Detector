import cv2
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cluster import KMeans
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font
import time

global cluster_centers_vertical_down, cluster_centers_vertical_top, cluster_centers_horizontal_left, cluster_centers_horizontal_right

def line_midpoint(line):
    x1, y1, x2, y2 = line
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def line_angle(line):
    x1, y1, x2, y2 = line
    slope = (y2 - y1) / (x2 - x1 + 1e-12)
    return abs(np.degrees(np.arctan(slope)))

def get_line_color(label, color_dict):
    """Returns consistent color for each cluster label."""
    if label not in color_dict:
        color_dict[label] = tuple(np.random.randint(0, 255, 3).tolist())
    return color_dict[label]

def generate_palette(n, base_hue):
    """Generate distinct RGB colors with different hue offsets"""
    colors = []
    for ii in range(n):
        hue = (base_hue + ii / n) % 1.0
        rgb = plt.cm.hsv(hue)
        colors.append(tuple(int(c*255) for c in rgb[:3]))
    return colors

def intersections_down_up(frame, x1, y1, m):
    """Computes the intersection with the two horizontal calibration lines"""
    x_down = (frame.shape[0] - y1 + m * x1) / m  # intersection with bottom border
    x_top = (0 - y1 + m * x1) / m # intersection with an horizontal line (above the bottom one)

    return x_down, x_top

def intersections_left_right(frame, x1, y1, m):
    """Computes the intersection with the two vertical (diagonal actually) calibration lines"""

    # Points for vertical lines (L and R)
    PL = (88, 88)
    PR = (635, 345)
    QL = (4, 357)
    QR = (525, 87)

    # Compute the m_left and m_right
    m_calibration_line_vertical_left = (PL[1]-QL[1]) / (PL[0]-QL[0])
    m_calibration_line_vertical_right = (PR[1]-QR[1]) / (PR[0]-QR[0])

    # Computation of intersection LEFT
    x_intersection_left = (m*x1 - y1 - m_calibration_line_vertical_left*PL[0] + PL[1]) / (m - m_calibration_line_vertical_left)
    y_left = m_calibration_line_vertical_left*(x_intersection_left - PL[0]) + PL[1]

    # Computation of intersection RIGHT
    x_intersection_right = (m*x1 - y1 - m_calibration_line_vertical_right*PR[0] + PR[1]) / (m - m_calibration_line_vertical_right)
    y_right = m_calibration_line_vertical_right*(x_intersection_right - PR[0]) + PR[1]

    # Draw the borders for the horizontal line intersection
    cv2.line(frame, (PL[0], PL[1]), (QL[0], QL[1]), (255, 0, 0), 7)
    cv2.line(frame, (PR[0], PR[1]), (QR[0], QR[1]), (255, 0, 0), 7)

    # return intersection values
    return y_left, y_right

# --- Parameters ---
MAX_TRACK_DIST = 1000      # maximum distance for considering two lines to be the same
MIN_ANGLE_VERTICAL = 80  # minimum angle to consider a line as vertical
cluster_centers_vertical_down = None
cluster_centers_horizontal_left = None
cluster_centers_vertical_top = None
cluster_centers_horizontal_right = None
vertical_colors = {}
horizontal_colors = {}
n_vertical = 7
n_horizontal = 7
index_calibration = 0

# Generate distinct color sets
vertical_palette = generate_palette(n_vertical, base_hue=0.003)
horizontal_palette = generate_palette(n_horizontal, base_hue=1.5)
print(np.size(vertical_palette))

# Tracking state
tracked_lines = []  # each element: {'id': int, 'line': (x1,y1,x2,y2)}
next_id = 0

# Load camera or video
cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("videos/test2.mp4")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Start time
start_time = time.time()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(blur, 50, 120)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=10)

    new_tracked = []
    if lines is not None:
        new_lines = [tuple(line[0]) for line in lines]
        if tracked_lines:
            cost_matrix = np.zeros((len(tracked_lines), len(new_lines)))
            for i, t in enumerate(tracked_lines):
                for j, l in enumerate(new_lines):
                    cost_matrix[i, j] = distance.euclidean(line_midpoint(t['line']), line_midpoint(l))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_new = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < MAX_TRACK_DIST:
                    tracked_lines[i]['line'] = new_lines[j]
                    new_tracked.append(tracked_lines[i])
                    matched_new.add(j)
            for j, l in enumerate(new_lines):
                if j not in matched_new:
                    new_tracked.append({'id': next_id, 'line': l})
                    next_id += 1
        else:
            for l in new_lines:
                new_tracked.append({'id': next_id, 'line': l})
                next_id += 1

        tracked_lines = new_tracked

    # Group the lines
    x_positions_down = []
    x_positions_top = []
    y_positions_left = []
    y_positions_right = []
    vertical_segments = []
    horizontal_segments = []

    for t in tracked_lines:
        x1, y1, x2, y2 = t['line']
        m = (y2 - y1) / (x2 - x1 + 1e-12)
        angle = line_angle(t['line'])

        if angle < 15:  # horizontal
            y_left, y_right = intersections_left_right(frame, x1, y1, m)
            y_positions_left.append(y_left)
            y_positions_right.append(y_right)
            horizontal_segments.append((x1, y1, x2, y2))
        else:  # vertical
            x_down, x_top = intersections_down_up(frame, x1, y1, m)
            x_positions_down.append(x_down)
            x_positions_top.append(x_top)
            vertical_segments.append((x1, y1, x2, y2))

            # For calibration of horizontal lines, for finding the left vertical line
            if x_down < 0 and index_calibration <1 and False:
                xL, yL, xQ, yQ = x1, y1, x2, x2
                print(f"Line: P1({xQ:.2f}, {yQ:.2f}), ({xL:.2f}, {yL:.2f})")
                index_calibration += 1

    # --- Clustering for the first frame down---
    if cluster_centers_vertical_down is None and len(x_positions_down) >= n_vertical:
        kmeans_v_down = KMeans(n_clusters=n_vertical, n_init=10)
        kmeans_v_down.fit(np.array(x_positions_down).reshape(-1, 1))
        cluster_centers_vertical_down = sorted(kmeans_v_down.cluster_centers_)
        # Assign stable colors
        for i in range(n_vertical):
            vertical_colors[i] = vertical_palette[i]
   
    # --- Clustering for the first frame up---
    if cluster_centers_vertical_top is None and len(x_positions_top) >= n_vertical:
        kmeans_v_top = KMeans(n_clusters=n_vertical, n_init=10)
        kmeans_v_top.fit(np.array(x_positions_top).reshape(-1, 1))
        cluster_centers_vertical_top = sorted(kmeans_v_top.cluster_centers_)

    # --- Clustering for the first frame left---
    if cluster_centers_horizontal_left is None and len(y_positions_left) >= n_horizontal:
        kmeans_h_left = KMeans(n_clusters=n_horizontal, n_init=10)
        kmeans_h_left.fit(np.array(y_positions_left).reshape(-1, 1))
        cluster_centers_horizontal_left = sorted(kmeans_h_left.cluster_centers_)
        for i in range(n_horizontal):
            horizontal_colors[i] = horizontal_palette[i]
   
    # --- Clustering for the first frame right---
    if cluster_centers_horizontal_right is None and len(y_positions_right) >= n_horizontal:
        kmeans_h_right = KMeans(n_clusters=n_horizontal, n_init=10)
        kmeans_h_right.fit(np.array(y_positions_right).reshape(-1, 1))
        cluster_centers_horizontal_right = sorted(kmeans_h_right.cluster_centers_)

   
    # Spectrum of horizontal lines
    # plt.figure("Left")
    # plt.scatter(np.ones(np.size(y_positions_left)), -1*np.ones(np.size(y_positions_left))*y_positions_left, c=kmeans_h_left.labels_)
    # plt.figure("Right")
    # plt.scatter(np.ones(np.size(y_positions_right)), -1*np.ones(np.size(y_positions_right))*y_positions_right, c=kmeans_h_right.labels_)
    # plt.show()
    # exit(0)

    # Spectrum of vertical lines
    # plt.figure("Down")
    # plt.scatter(x_positions_down, np.ones(np.size(x_positions_down)) , c=kmeans_v_down.labels_)
    # plt.figure("Top")
    # plt.scatter(x_positions_top, np.ones(np.size(x_positions_top)) , c=kmeans_v_top.labels_)
    # plt.show()
    # exit(0)

    # --- Assign each new line to the nearest cluster ---
    red_lines = 0
    if cluster_centers_vertical_down is not None and cluster_centers_vertical_top is not None:
        for line in vertical_segments:
            x1, y1, x2, y2 = line
            m = (y2 - y1) / (x2 - x1 + 1e-12)
            x_down, x_top = intersections_down_up(frame, x1, y1, m)

            # Find closest cluster
            idx_down = np.argmin(distance.cdist([[x_down]], cluster_centers_vertical_down))
            idx_top = np.argmin(distance.cdist([[x_top]], cluster_centers_vertical_top))
            if idx_down == idx_top:
                color = (0, 255, 0) # green
            else:
                color = (0, 0, 255) # red
                red_lines += 1
            # color = vertical_colors[idx_down]
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    if cluster_centers_horizontal_left is not None and cluster_centers_horizontal_right is not None:
        for line in horizontal_segments:
            x1, y1, x2, y2 = line
            m = (y2 - y1) / (x2 - x1 + 1e-12)
            y_left, y_right = intersections_left_right(frame, x1, y1, m)
            idx_left = np.argmin(distance.cdist([[y_left]], cluster_centers_horizontal_left))
            idx_right = np.argmin(distance.cdist([[y_right]], cluster_centers_horizontal_right))
           
            if idx_left == idx_right:
                color = (0, 255, 0) # green
            else:
                red_lines += 1
                color = (0, 0, 255) # red
            # color = horizontal_colors[idx_left]
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    print("Red lines= ", red_lines)
    cv2.imshow("Edges Detector", frame)
    cv2.imshow("Edges with Canny", edges)

    # Compare the current time
    now = time.time()-start_time
    simulation_time = 300

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q') or red_lines > 15000 or now > simulation_time: #15
        break

# Close all the windows
cap.release()
cv2.destroyAllWindows()