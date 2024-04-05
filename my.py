import cv2
import numpy as np
from collections import OrderedDict

# Constants
PIXEL_TO_METERS = 0.1  # Pixel scaling factor (for converting pixels to meters)
FPS = 30  # Frame rate of the video
MIN_WIDTH_RECT = 80 
MIN_HEIGHT_RECT = 80
COUNT_LINE_POSITION = 550
COUNT_LINE_POSITION_EXIT = 800
OFFSET = 6

# Centroid tracker class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        objectID = str(len(self.objects) + 1)
        self.objects[objectID] = (centroid, None)
        self.disappeared[objectID] = 0

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[objID][0] for objID in objectIDs]

            D = np.zeros((len(objectIDs), len(inputCentroids)))

            for i in range(len(objectIDs)):
                for j in range(len(inputCentroids)):
                    D[i, j] = np.linalg.norm(objectCentroids[i] - inputCentroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = (inputCentroids[col], objectCentroids[row])
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')

# Initialize background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Initialize centroid tracker
ct = CentroidTracker()

# Dictionary to store tracks (vehicle ID to centroid coordinates, previous centroid, and speed)
tracks = OrderedDict()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction
    img_sub = algo.apply(blur)

    # Morphological operations to enhance detection
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    dilatadata = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, np.ones((5, 5)))
    dilatadata = cv2.morphologyEx(dilatadata, cv2.MORPH_CLOSE, np.ones((5, 5)))

    # Find contours of detected objects
    contours, _ = cv2.findContours(dilatadata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        validate_counter = (w >= MIN_WIDTH_RECT) and (h >= MIN_HEIGHT_RECT)
        if not validate_counter:
            continue
        rects.append((x, y, x + w, y + h))

    objects = ct.update(rects)


    vehicles_between = 0
    for objectID, (centroid, _) in objects.items():
        if centroid[1] < (COUNT_LINE_POSITION_EXIT + OFFSET) and centroid[1] > (COUNT_LINE_POSITION - OFFSET):
            vehicles_between += 1

    JAM_THRESHOLD = 0
    jam = vehicles_between >= JAM_THRESHOLD


    for objectID, (centroid, prevCentroid) in objects.items():
        cv2.putText(frame, f"Vehicle {objectID}", (centroid[0], centroid[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        if prevCentroid is not None:
            dx = (centroid[0] - prevCentroid[0]) * PIXEL_TO_METERS
            dy = (centroid[1] - prevCentroid[1]) * PIXEL_TO_METERS
            speed = np.sqrt(dx**2 + dy**2) * FPS  # Speed in meters per second

            cv2.putText(frame, f"Speed: {speed:.2f} m/s", (centroid[0], centroid[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        if centroid[1] < (COUNT_LINE_POSITION + OFFSET) and centroid[1] > (COUNT_LINE_POSITION - OFFSET):
            cv2.line(frame, (25, COUNT_LINE_POSITION), (1200, COUNT_LINE_POSITION), (0, 127, 255), 3)
            print("Vehicle counter:", len(objects))

        if centroid[1] < (COUNT_LINE_POSITION_EXIT + OFFSET) and centroid[1] > (COUNT_LINE_POSITION_EXIT - OFFSET):
            cv2.line(frame, (25, COUNT_LINE_POSITION_EXIT), (1200, COUNT_LINE_POSITION_EXIT), (0, 127, 255), 3)
            print("Vehicle counter:", len(objects))

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break
cap.release()
cv2.destroyAllWindows()

