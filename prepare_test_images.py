# Adapted from Haiyu Wu's prepare_test_images.py:
# https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test/blob/main/utils/prepare_test_images.py
# Modified to build test sets from gen/imp folder structure instead of .xz files.

from tqdm import tqdm
import argparse
import numpy as np
from os import path, makedirs
from pathlib import Path
import cv2

NUM_FOLDS = 10
TWINS_PER_FOLD = 5
POSITIVE_PER_PAIR = 132   # C(12,2) * 2
NEGATIVE_PER_PAIR = 132   # sampled from 144
GEN_PER_FOLD = TWINS_PER_FOLD * POSITIVE_PER_PAIR   # 660
IMP_PER_FOLD = TWINS_PER_FOLD * NEGATIVE_PER_PAIR   # 660
PAIRS_PER_FOLD = GEN_PER_FOLD + IMP_PER_FOLD        # 1320


# Read, resize, and convert a single image
def read_image(image_path, image_size):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return None
    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Load images from gen/imp folder structure in fold order.
# Folder numbering (from pos_neg.py):
#   gen/: 0000 to (NUM_FOLDS * GEN_PER_FOLD - 1)  →  0000 to 6599
#   imp/: 0000 to (NUM_FOLDS * IMP_PER_FOLD - 1)  →  0000 to 6599
# Output order per fold:
#   gen_fold_0 (660 pairs), imp_fold_0 (660 pairs),
#   gen_fold_1 (660 pairs), imp_fold_1 (660 pairs), ...
def load_from_folders(pairs_folder, image_size=(112, 112)):
    pairs_folder = Path(pairs_folder)

    gen_folder = pairs_folder / 'gen'
    imp_folder = pairs_folder / 'imp'

    gen_dirs = sorted(gen_folder.iterdir(), key=lambda x: int(x.name))
    imp_dirs = sorted(imp_folder.iterdir(), key=lambda x: int(x.name))

    print(f"Total gen folders: {len(gen_dirs)} (expected {NUM_FOLDS * GEN_PER_FOLD})")
    print(f"Total imp folders: {len(imp_dirs)} (expected {NUM_FOLDS * IMP_PER_FOLD})")

    if len(gen_dirs) != NUM_FOLDS * GEN_PER_FOLD:
        print(f"gen folder count mismatch")
    if len(imp_dirs) != NUM_FOLDS * IMP_PER_FOLD:
        print(f"imp folder count mismatch")

    imgs = []
    issame = []
    skipped = 0

    for fold in range(NUM_FOLDS):
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")

        gen_start = fold * GEN_PER_FOLD
        gen_end   = gen_start + GEN_PER_FOLD
        imp_start = fold * IMP_PER_FOLD
        imp_end   = imp_start + IMP_PER_FOLD

        # Load gen pairs for this fold
        for pair_dir in tqdm(gen_dirs[gen_start:gen_end], desc=f"Fold {fold+1} gen"):
            images = sorted(pair_dir.glob('*.jpg'))
            if len(images) != 2:
                print(f"Skipping {pair_dir}: only {len(images)} images")
                skipped += 1
                continue
            img1 = read_image(images[0], image_size)
            img2 = read_image(images[1], image_size)
            if img1 is None or img2 is None:
                skipped += 1
                continue
            imgs.append(img1)
            imgs.append(img2)
            issame.append(True)

        # Load imp pairs for this fold
        for pair_dir in tqdm(imp_dirs[imp_start:imp_end], desc=f"Fold {fold+1} imp"):
            images = sorted(pair_dir.glob('*.jpg'))
            if len(images) < 2:
                print(f"Skipping {pair_dir}: only {len(images)} images")
                skipped += 1
                continue
            img1 = read_image(images[0], image_size)
            img2 = read_image(images[1], image_size)
            if img1 is None or img2 is None:
                skipped += 1
                continue
            imgs.append(img1)
            imgs.append(img2)
            issame.append(False)

    if skipped > 0:
        print(f"\nTotal skipped pairs: {skipped}")

    # Shape: (N*2, H, W, C) → (N*2, C, H, W)
    dataset = np.asarray(imgs).transpose((0, 3, 1, 2))
    return dataset, issame


def convert_to_test(images, issame, dataset_name, destination):
    save_folder = f"{destination}/{dataset_name}"
    if not path.exists(save_folder):
        makedirs(save_folder)

    np.save(f"{save_folder}/{dataset_name}.npy", images)
    np.savetxt(f"{save_folder}/issame.txt", np.array(issame).astype(int), fmt="%s")
    print(f"\nSaved {len(issame)} pairs to {save_folder}")

    # Verify fold structure
    print("\nVerifying fold structure in issame.txt:")
    issame_array = np.array(issame).astype(int)
    all_valid = True
    for fold_num in range(NUM_FOLDS):
        start_idx = fold_num * PAIRS_PER_FOLD
        end_idx   = start_idx + PAIRS_PER_FOLD
        fold_labels = issame_array[start_idx:end_idx]
        num_positive = int(np.sum(fold_labels))
        num_negative = len(fold_labels) - num_positive
        valid = num_positive == GEN_PER_FOLD and num_negative == IMP_PER_FOLD
        flag = "✓" if valid else "✗"
        print(f"  Fold {fold_num}: {num_positive} genuine, {num_negative} impostor "
              f"(total {len(fold_labels)}) {flag}")
        if not valid:
            all_valid = False

    if all_valid:
        print("\nAll folds verified successfully")
    else:
        print("\nSome folds have unexpected counts")


def main(args):
    print(f"Processing dataset: {args.dataset_name}")
    print(f"Source folder:      {args.pairs_folder}")
    print(f"Destination:        {args.destination}")
    print(f"\nExpected structure:")
    print(f"  Folds:            {NUM_FOLDS}")
    print(f"  Twins per fold:   {TWINS_PER_FOLD}")
    print(f"  Gen per fold:     {GEN_PER_FOLD}")
    print(f"  Imp per fold:     {IMP_PER_FOLD}")
    print(f"  Pairs per fold:   {PAIRS_PER_FOLD}")
    print(f"  Total pairs:      {NUM_FOLDS * PAIRS_PER_FOLD}")

    images, issame = load_from_folders(args.pairs_folder, image_size=(112, 112))

    print(f"\nLoaded {len(images)} images ({len(issame)} pairs)")
    print(f"  Positive pairs: {sum(issame)}")
    print(f"  Negative pairs: {len(issame) - sum(issame)}")
    print(f"  Image array shape: {images.shape}")

    convert_to_test(images, issame, args.dataset_name, args.destination)

    print(f"\nCreated:")
    print(f"  {args.destination}/{args.dataset_name}/{args.dataset_name}.npy")
    print(f"  {args.destination}/{args.dataset_name}/issame.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert gen/imp folders to .npy format')
    parser.add_argument('--pairs_folder', '-f', type=str, required=True,
                        help='folder containing gen/ and imp/ subdirectories')
    parser.add_argument('--destination', '-d', type=str, default='./test_sets',
                        help='destination folder')
    parser.add_argument('--dataset_name', '-n', type=str, required=True,
                        help='name of the dataset (e.g., twins)')
    args = parser.parse_args()
    main(args)