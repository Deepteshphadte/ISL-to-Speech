import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mediapipe_utils import collect_data

# rest of the code stays the same...


LABELS_FILE = "data/labels.txt"
DATASET_PATH = "data/processed"


def load_labels():

    with open(LABELS_FILE, "r") as file:

        labels = [

            line.strip().upper()

            for line in file

            if line.strip()

        ]

    return labels

def show_dataset_status():

    labels = load_labels()

    print("\n" + "=" * 65)
    print("                 DATASET STATUS")
    print("=" * 65)

    completed = 0

    for label in labels:

        folder = os.path.join(
            DATASET_PATH,
            label
        )

        if os.path.exists(folder):

            count = len([
                file
                for file in os.listdir(folder)
                if file.endswith(".npy")
            ])

        else:

            count = 0

        if count >= 200:

            status = "✅ Complete"
            completed += 1

        elif count >= 100:

            status = "⚠ Good"

        elif count > 0:

            status = "❌ Need More"

        else:

            status = "⭕ Missing"

        print(
            f"{label:<20}"
            f"{count:<10}"
            f"{status}"
        )

    print("=" * 65)

    print(f"Total Labels : {len(labels)}")
    print(f"Completed    : {completed}")
    print(f"Remaining    : {len(labels)-completed}")


def print_menu():

    labels = load_labels()

    print("\n" + "=" * 60)
    print("         ISL DATA COLLECTION TOOL")
    print("=" * 60)
    print("STATUS -> View Dataset Progress")

    print(f"\nTotal Supported Labels : {len(labels)}")

    print("\nCommands")
    print("LIST  -> Show all labels")
    print("STATUS -> View Dataset Progress")
    print("EXIT  -> Quit")

    print("=" * 60)


def print_all_labels():

    labels = load_labels()

    print("\nAvailable Labels\n")

    for i, label in enumerate(labels, start=1):

        print(f"{i:3}. {label}")


if __name__ == "__main__":
    print_menu()

    ALL_LABELS = load_labels()

    while True:
        label = input("\nEnter label: ").strip().upper()

        if label == "EXIT":
            print("Exiting data collection. Goodbye!")
            break

        elif label == "LIST":
            print_all_labels()

        elif label == "STATUS":

            show_dataset_status()

        elif label in ALL_LABELS:
            print(f"\nStarting collection for: {label}")
            print("Press 's' to start recording a sample, 'q' to go back to menu.")
            collect_data(label=label)

        else:
            print(f"[ERROR] '{label}' is not a valid label.")
            print("Type 'LIST' to see all valid labels.")