# Adapted from Haiyu Wu's prepare_test_images.py:
# https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test/blob/main/utils/prepare_test_images.py
# Modified to load image pairs from gen/imp folders instead of .xz files.

from tqdm import tqdm
import argparse
import numpy as np
from os import path, makedirs
from pathlib import Path
import cv2


NUM_FOLDS = 10
TWINS_PER_FOLD = 8

POSITIVE_PER_PAIR = 132
NEGATIVE_PER_PAIR = 132

GEN_PER_FOLD = TWINS_PER_FOLD * POSITIVE_PER_PAIR
IMP_PER_FOLD = TWINS_PER_FOLD * NEGATIVE_PER_PAIR
PAIRS_PER_FOLD = GEN_PER_FOLD + IMP_PER_FOLD


def read_image(image_path, image_size):
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Could not read {image_path}")
        return None

    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_from_folders(pairs_folder, image_size=(112, 112)):
    pairs_folder = Path(pairs_folder)

    gen_folder = pairs_folder / "gen"
    imp_folder = pairs_folder / "imp"

    gen_dirs = []
    for pair_dir in gen_folder.iterdir():
        if pair_dir.is_dir():
            gen_dirs.append(pair_dir)

    imp_dirs = []
    for pair_dir in imp_folder.iterdir():
        if pair_dir.is_dir():
            imp_dirs.append(pair_dir)

    gen_dirs.sort(key=lambda folder: int(folder.name))
    imp_dirs.sort(key=lambda folder: int(folder.name))

    expected_gen = NUM_FOLDS * GEN_PER_FOLD
    expected_imp = NUM_FOLDS * IMP_PER_FOLD

    print(f"Gen folders: {len(gen_dirs)}")
    print(f"Imp folders: {len(imp_dirs)}")

    if len(gen_dirs) != expected_gen:
        print(f"Expected {expected_gen} gen folders")

    if len(imp_dirs) != expected_imp:
        print(f"Expected {expected_imp} imp folders")

    imgs = []
    issame = []
    skipped = 0

    for fold in range(NUM_FOLDS):
        gen_start = fold * GEN_PER_FOLD
        gen_end = gen_start + GEN_PER_FOLD

        imp_start = fold * IMP_PER_FOLD
        imp_end = imp_start + IMP_PER_FOLD

        gen_fold_dirs = gen_dirs[gen_start:gen_end]

        for pair_dir in tqdm(gen_fold_dirs, desc=f"Fold {fold + 1} gen"):
            images = list(pair_dir.glob("*.jpg"))
            images.sort()

            if len(images) != 2:
                print(f"Skipping {pair_dir}: found {len(images)} images")
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

        imp_fold_dirs = imp_dirs[imp_start:imp_end]

        for pair_dir in tqdm(imp_fold_dirs, desc=f"Fold {fold + 1} imp"):
            images = list(pair_dir.glob("*.jpg"))
            images.sort()

            if len(images) != 2:
                print(f"Skipping {pair_dir}: found {len(images)} images")
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
        print(f"Skipped pairs: {skipped}")

    dataset = np.asarray(imgs)
    dataset = dataset.transpose((0, 3, 1, 2))

    return dataset, issame


def convert_to_test(images, issame, dataset, destination):
    save_folder = f"{destination}/{dataset}"

    if not path.exists(save_folder):
        makedirs(save_folder)

    np.save(f"{save_folder}/{dataset}.npy", images)

    labels = np.array(issame).astype(int)
    np.savetxt(f"{save_folder}/issame.txt", labels, fmt="%s")

    all_valid = True

    for fold_num in range(NUM_FOLDS):
        start_idx = fold_num * PAIRS_PER_FOLD
        end_idx = start_idx + PAIRS_PER_FOLD

        fold_labels = labels[start_idx:end_idx]
        num_positive = int(np.sum(fold_labels))
        num_negative = len(fold_labels) - num_positive

        if num_positive != GEN_PER_FOLD:
            all_valid = False

        if num_negative != IMP_PER_FOLD:
            all_valid = False

        print(
            f"Fold {fold_num}: {num_positive} genuine, "
            f"{num_negative} impostor"
        )

    if all_valid:
        print("Fold counts are correct")
    else:
        print("Some fold counts are incorrect")


def main(args):
    images, issame = load_from_folders(
        args.pairs_folder,
        image_size=(112, 112)
    )

    print(f"\nImages: {len(images)}")
    print(f"Pairs: {len(issame)}")
    print(f"Positive pairs: {sum(issame)}")
    print(f"Negative pairs: {len(issame) - sum(issame)}")
    print(f"Image shape: {images.shape}")
    print()

    convert_to_test(
        images,
        issame,
        args.dataset_name,
        args.destination
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert gen/imp folders to .npy format"
    )

    parser.add_argument(
        "--pairs_folder",
        "-f",
        type=str,
        required=True,
        help="folder containing gen and imp folders"
    )

    parser.add_argument(
        "--destination",
        "-d",
        type=str,
        default="./test_sets",
        help="destination folder"
    )

    parser.add_argument(
        "--dataset_name",
        "-n",
        type=str,
        required=True,
        help="dataset name"
    )

    args = parser.parse_args()
    main(args)