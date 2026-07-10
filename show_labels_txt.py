LABELS_FILE = "data/labels.txt"

with open(LABELS_FILE, "r") as f:

    labels = [

        line.strip()

        for line in f

        if line.strip()

    ]

print()

print("Total Labels :", len(labels))

print()

for i, label in enumerate(labels):

    print(f"{i:03d}  {label}")