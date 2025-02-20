import random
import json
import csv
import os


TRAIN_SIZE = 0.8
BLACKLIST = []

CWD = os.path.dirname(os.path.realpath(__file__))

DATASET_FOLDER = os.path.join(CWD, "data", "dataset")

RIGHT_TRAIN_FILEPATH = os.path.join(CWD, "data", "right_train.csv")
RIGHT_TEST_FILEPATH = os.path.join(CWD, "data", "right_test.csv")
LEFT_TRAIN_FILEPATH = os.path.join(CWD, "data", "left_train.csv")
LEFT_TEST_FILEPATH = os.path.join(CWD, "data", "left_test.csv")

LABEL_FILEPATH = os.path.join(CWD, "data", "labels.json")

LABELS = []


def clear_files():
    with open(RIGHT_TRAIN_FILEPATH, "w"):
        pass
    with open(RIGHT_TEST_FILEPATH, "w"):
        pass
    with open(LEFT_TRAIN_FILEPATH, "w"):
        pass
    with open(LEFT_TEST_FILEPATH, "w"):
        pass


def collect_labels() -> list[str]:
    labels = []
    for label in os.listdir(DATASET_FOLDER):
        if label in BLACKLIST:
            continue

        label_folder_path = os.path.join(DATASET_FOLDER, label)

        if os.path.isfile(label_folder_path):
            continue

        if len(os.listdir(label_folder_path)) > 0:
            labels.append(label)
    return labels


def write_data_for_label(label: str):
    label_folder_path = os.path.join(DATASET_FOLDER, label)
    current_label_id = LABELS.index(label)

    for handedness in os.listdir(label_folder_path):
        current_data_filepath = os.path.join(label_folder_path, handedness)

        if not list(csv.reader(open(current_data_filepath, "r"))):
            continue

        current_train_path = (
            LEFT_TRAIN_FILEPATH if handedness == "left.csv" else RIGHT_TRAIN_FILEPATH
        )
        current_test_path = (
            LEFT_TEST_FILEPATH if handedness == "left.csv" else RIGHT_TEST_FILEPATH
        )

        train_file = open(current_train_path, "a", newline="")
        test_file = open(current_test_path, "a", newline="")

        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        print(
            f"Adding {current_data_filepath} to the {current_test_path} and {current_train_path} datasets..."
        )
        with open(current_data_filepath, "r") as current_data_file:
            rows = list(csv.reader(current_data_file))

            train_data = rows[: int(len(rows) * TRAIN_SIZE)]
            test_data = rows[int(len(rows) * TRAIN_SIZE) :]

            for row in train_data:
                train_writer.writerow([current_label_id] + row)
            for row in test_data:
                test_writer.writerow([current_label_id] + row)


def shuffle_csv(file_path, shuffle_path):
    with open(file_path, "r") as dataset_file:
        rows = list(csv.reader(dataset_file))
        random.shuffle(rows)
        with open(shuffle_path, "w", newline="") as dataset_shuffle_file:
            writer = csv.writer(dataset_shuffle_file)
            writer.writerows(rows)


def main():
    global LABELS
    LABELS = collect_labels()

    # Write labels to file
    with open(LABEL_FILEPATH, "w") as label_file:
        json.dump(LABELS, label_file)

    clear_files()
    for label in LABELS:
        write_data_for_label(label)

    # shuffle_csv(RIGHT_TRAIN_FILEPATH, RIGHT_TRAIN_FILEPATH)
    # shuffle_csv(RIGHT_TEST_FILEPATH, RIGHT_TEST_FILEPATH)
    # shuffle_csv(LEFT_TRAIN_FILEPATH, LEFT_TRAIN_FILEPATH)
    # shuffle_csv(LEFT_TEST_FILEPATH, LEFT_TEST_FILEPATH)

    print("Data aggregation complete.")


if __name__ == "__main__":
    main()
