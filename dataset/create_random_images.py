import json
import random

"""
This script is used for selecting 100 images randomly from the complete FSC147 dataset
In order to test the robustness, randomly selected 100 images should well reflect its performance
And 100 images is all our hardware can spare ///...///
"""

SPLIT_FILE = "Train_Test_Val_FSC_147.json"
ANN_FILE = "annotation_FSC147_384.json"
OUTPUT_FILE = "fsc147_test_100_stratified.txt"

random.seed(42)  # reproducibility

NUM_TOTAL = 100
N_LOW = 30    # ≤20 objects
N_MID = 40    # 21–50 objects
N_HIGH = 30   # >50 objects

LOW_MAX = 20
MID_MAX = 50


def main():
    #Load split file
    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    test_imgs = splits["test"]
    print(f"Total images in test split: {len(test_imgs)}")

    #Load annotation file
    with open(ANN_FILE, "r") as f:
        ann = json.load(f)

    #Collect (img_name, count)
    infos = []
    for img_name in test_imgs:
        if img_name not in ann:
            continue
        gt_count = len(ann[img_name]["points"])
        infos.append((img_name, gt_count))

    print(f"Test images with annotation: {len(infos)}")

    #Stratify by difficulty
    low, mid, high = [], [], []
    for img_name, gt in infos:
        if gt <= LOW_MAX:
            low.append((img_name, gt))
        elif gt <= MID_MAX:
            mid.append((img_name, gt))
        else:
            high.append((img_name, gt))

    print(f"Low count ≤{LOW_MAX}: {len(low)}")
    print(f"Mid count {LOW_MAX+1}-{MID_MAX}: {len(mid)}")
    print(f"High count >{MID_MAX}: {len(high)}")

    # Sample from each bin
    selected = []
    selected += random.sample(low, min(len(low), N_LOW))
    selected += random.sample(mid, min(len(mid), N_MID))
    selected += random.sample(high, min(len(high), N_HIGH))

    #Top-up(ok, i am not talking about gaming industry) if any category lacked enough images
    if len(selected) < NUM_TOTAL:
        selected_names = set(img for img, _ in selected)
        remaining = [(img, c) for img, c in infos if img not in selected_names]
        needed = NUM_TOTAL - len(selected)
        selected += random.sample(remaining, min(needed, len(remaining)))

    selected_names = [img for img, _ in selected]

    # Save output file
    with open(OUTPUT_FILE, "w") as f:
        for name in selected_names:
            f.write(name + "\n")

    print("-" * 50)
    print(f"Saved {len(selected_names)} image names → {OUTPUT_FILE}")
    print("Subset created successfully!")


if __name__ == "__main__":
    main()
