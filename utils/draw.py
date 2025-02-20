import cv2


CONNECTIONS = [
    [4, 3, 2, 1, 0, 5, 9, 13, 17, 0],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]


def draw_hand(image, bbox, keypoints, thickness=2, colors=((255, 255, 255), (0, 0, 0))):
    # Keypoints
    for i, point in enumerate(keypoints):
        cv2.circle(image, tuple(point), 3, colors[0], thickness * 2)

    # Connections
    for path in CONNECTIONS:
        for i in range(len(path) - 1):
            cv2.line(
                image,
                tuple(keypoints[path[i]]),
                tuple(keypoints[path[i + 1]]),
                colors[0],
                thickness,
            )

    # Bounding box
    cv2.rectangle(
        image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[0], thickness * 2
    )

    return image


def draw_info(image, bbox, handedness, gesture, confidence, colors=((255, 255, 255), (0, 0, 0))):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[1] - 22), colors[0], -1)

    hand_text = handedness
    if gesture != "":
        hand_text += ": " + gesture + " (" + str(round(confidence * 100)) + "%)"
    cv2.putText(
        image,
        hand_text,
        (bbox[0] + 5, bbox[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        colors[1],
        2,
    )

    return image
