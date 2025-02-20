from model import GestureClassifier
from utils import algo, draw
import mediapipe as mp
import json
import cv2


SUCCESS_COLORS = ((0, 255, 0), (0, 0, 0))
FAILURE_COLORS = ((0, 0, 255), (0, 0, 0))


class Engine:
    def __init__(
        self,
        labels: list[str],
        model_left_path: str,
        model_right_path: str,
        threshold=0.7,
    ):
        self.labels = labels
        self.model_left_path = model_left_path
        self.model_right_path = model_right_path
        self.threshold = threshold

        self.ready = False

    def prepare(self):
        # Camera preparation ##################################################
        print("Prepering camera...")
        self._cap = cv2.VideoCapture(0)
        print("Done!")

        # Model load ##########################################################
        print("Loading hand landmark model...")
        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=8,
        )
        print("Done!")

        print("Loading gesture classifier model...")
        self._gesture_classifier_left = GestureClassifier(self.model_left_path)
        self._gesture_classifier_right = GestureClassifier(self.model_right_path)
        print("Done!")

        self.ready = True

    def estimate_hand_landmarks(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self._hands.process(image)
        image.flags.writeable = True
        return results

    def process_hand_landmarks(self, image, multi_hand_landmarks, multi_handedness):
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            handedness = handedness.classification[0].label[0:]

            keypoints = algo.calc_keypoints(image, hand_landmarks)
            bbox = algo.calc_bounding_box(keypoints)

            # Conversion to relative coordinates - normalized coordinates
            normalized_keypoints = algo.normalize_keypoints(keypoints)

            # Gesture classification
            if handedness == "Right":
                gesture_id, conf = self._gesture_classifier_right(normalized_keypoints)
            else:
                gesture_id, conf = self._gesture_classifier_left(normalized_keypoints)

            # Draw
            if conf < self.threshold:
                continue
            colors = ((0, int(max(255, 400 * conf)), int(min(255, 400 * (1 - conf)))), (0, 0, 0))
            draw.draw_hand(image, bbox, keypoints, 2, colors)
            draw.draw_info(
                image, bbox, handedness, self.labels[gesture_id], conf, colors
            )

    def __call__(self):
        if not self.ready:
            self.prepare()

        # Parse mode (ESC to exit) ########################################
        key = cv2.waitKey(20)
        if key == 27:
            return None

        # Camera capture ##################################################
        ret, image = self._cap.read()
        if not ret:
            return None
        image = cv2.flip(image, 1)

        # Process hands ###################################################
        results = self.estimate_hand_landmarks(image)
        if results.multi_hand_landmarks is not None:
            self.process_hand_landmarks(
                image, results.multi_hand_landmarks, results.multi_handedness
            )

        # Debug window ####################################################
        cv2.imshow("Debug", image)

        return image


def main():
    labels = json.load(open("data/labels.json"))
    model_left_path = "./models/gesture_classifier_left.tflite"
    model_right_path = "./models/gesture_classifier_right.tflite"
    threshold = 0.2

    engine = Engine(labels, model_left_path, model_right_path, threshold)

    while True:
        result = engine()
        if result is None:
            break


if __name__ == "__main__":
    main()
