from utils.mediapipe_utils import collect_data

if __name__ == "__main__":
    while True:
        label = input("Enter label (A/B/C or 'exit'): ").upper()

        if label == "EXIT":
            break

        collect_data(label=label)