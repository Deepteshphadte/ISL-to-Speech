import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mediapipe_utils import collect_data

# rest of the code stays the same...

# ─── All supported gestures ───────────────────────────────────────────────────
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]          # A-Z
NUMBERS = [str(i) for i in range(1, 10)]                            # 1-9
WORDS   = [
    "HELLO", "GOOD", "BAD", "YES", "NO",
    "THANK_YOU", "SORRY", "HELP", "PLEASE", "I_LOVE_YOU"
]
SPECIAL = ["NONE"]

ALL_LABELS = LETTERS + NUMBERS + WORDS + SPECIAL
# ──────────────────────────────────────────────────────────────────────────────


def print_menu():
    print("\n" + "="*55)
    print("       ISL DATA COLLECTION TOOL")
    print("="*55)
    print("Letters  : A - Z")
    print("Numbers  : 1 - 9")
    print("Words    : HELLO, GOOD, BAD, YES, NO,")
    print("           THANK_YOU, SORRY, HELP, PLEASE, I_LOVE_YOU")
    print("Special  : NONE  (no gesture / background)")
    print("-"*55)
    print("Type the label to collect data for it.")
    print("Type 'LIST' to see all labels.")
    print("Type 'EXIT' to quit.")
    print("="*55)


def print_all_labels():
    print("\nAll supported labels:")
    print(f"  Letters : {' '.join(LETTERS)}")
    print(f"  Numbers : {' '.join(NUMBERS)}")
    print(f"  Words   : {' '.join(WORDS)}")
    print(f"  Special : {' '.join(SPECIAL)}")


if __name__ == "__main__":
    print_menu()

    while True:
        label = input("\nEnter label: ").strip().upper()

        if label == "EXIT":
            print("Exiting data collection. Goodbye!")
            break

        elif label == "LIST":
            print_all_labels()

        elif label in ALL_LABELS:
            print(f"\nStarting collection for: {label}")
            print("Press 's' to start recording a sample, 'q' to go back to menu.")
            collect_data(label=label)

        else:
            print(f"[ERROR] '{label}' is not a valid label.")
            print("Type 'LIST' to see all valid labels.")