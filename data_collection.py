import os


CWD = os.path.dirname(os.path.realpath(__file__))
LABEL_NAMES = os.listdir(os.path.join(CWD, "data", "dataset"))

# Print the avaiable labels
[print(f"{str(i).ljust(2)} : {label}") for i, label in enumerate(LABEL_NAMES)]

# Get the label ID from the user
LABEL_ID = ""
while True:
    LABEL_ID = input("Enter the label ID: ")
    try:
        LABEL_ID = int(LABEL_ID)
        if LABEL_ID < len(LABEL_NAMES):
            break
    except ValueError:
        print("Invalid label ID. Please try again.")
print(f"Selected label: {LABEL_NAMES[LABEL_ID]}")

# Create the data folder path
DATA_FOLDER = os.path.join(CWD, "data", "dataset", LABEL_NAMES[LABEL_ID])
LEFT_HAND_DATA_FILE_PATH = os.path.join(DATA_FOLDER, "left.csv")
RIGHT_HAND_DATA_FILE_PATH = os.path.join(DATA_FOLDER, "right.csv")


print("Importing libraries...")
from utils import algo, draw
import mediapipe as mp
import cv2
import csv

# Initialize the MediaPipe Hands model
print("Initializing MediaPipe Hands model...")
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
print("MediaPipe Hands model initialized.")

# Initialize the webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)
print("Webcam initialized.")

left_lines = 0 if not os.path.exists(LEFT_HAND_DATA_FILE_PATH) else sum(1 for _ in open(LEFT_HAND_DATA_FILE_PATH))
right_lines = 0 if not os.path.exists(RIGHT_HAND_DATA_FILE_PATH) else sum(1 for _ in open(RIGHT_HAND_DATA_FILE_PATH))
count = [left_lines, right_lines]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks is not None:
        landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        keypoints = algo.calc_keypoints(frame, landmarks)
        bbox = algo.calc_bounding_box(keypoints)
        normalized_keypoints = algo.normalize_keypoints(keypoints)

        draw.draw_hand(frame, bbox, keypoints)
        draw.draw_info(frame, bbox, handedness, "", "")

    key = cv2.waitKey(1)
    if key == 32:
        file_to_open = LEFT_HAND_DATA_FILE_PATH if handedness == "Left" else RIGHT_HAND_DATA_FILE_PATH
        idx = 0 if handedness == "Left" else 1
        count[idx] += 1

        with open(file_to_open, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(normalized_keypoints)
        print(f"[{handedness} {count[idx]}] Data saved to {file_to_open}")
    elif key == 27:
        print("Data collection stopped.")
        break

    cv2.imshow("Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
