import cv2
import numpy as np
import random

cap = cv2.VideoCapture("in.avi")
human_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

line_position = 250
next_object_id = 0
trackers = {}
ages = {}

def get_centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.1, 2)

    current_centroids = []
    rects = []

    for (x, y, w, h) in humans:
        centroid = get_centroid(x, y, w, h)
        current_centroids.append(centroid)
        rects.append((x, y, w, h))

    updated_trackers = {}

    for i, centroid in enumerate(current_centroids):
        matched_id = None
        for object_id, prev_centroid in trackers.items():
            if euclidean_distance(centroid, prev_centroid) < 50:
                matched_id = object_id
                break

        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1
            ages[matched_id] = random.randint(5, 80) 

        updated_trackers[matched_id] = centroid

        x, y, w, h = rects[i]
        age = ages[matched_id]

       
        color = (0, 0, 255) if age > 45 else (0, 255, 0)
        label = f"Age: {age}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, centroid, 4, color, -1)

    trackers = updated_trackers

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)

    cv2.imshow("People with Age Highlight", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()