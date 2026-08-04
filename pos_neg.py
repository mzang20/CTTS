import argparse
import json
import random
import shutil
from pathlib import Path


TWINS_PER_FOLD = 8
IMAGES_PER_TWIN = 12
POSITIVE_PER_TWIN = 66
POSITIVE_PER_PAIR = 132
NEGATIVE_PER_PAIR = 132
PAIRS_PER_TWIN_PAIR = 264
NUM_FOLDS = 10


def collect_twins(celeb_twins_path):
    print("Collecting twins")

    all_twins = []

    fold_directories = list(celeb_twins_path.iterdir())
    fold_directories.sort()

    for fold_dir in fold_directories:
        if not fold_dir.is_dir():
            continue

        if not fold_dir.name.startswith("fold_"):
            continue

        for twin_pair_folder in fold_dir.iterdir():
            if not twin_pair_folder.is_dir():
                continue

            twin_folders = []

            for folder in twin_pair_folder.iterdir():
                if folder.is_dir():
                    twin_folders.append(folder)

            twin_folders.sort()

            for twin_folder in twin_folders:
                images = list(twin_folder.glob("*.jpg"))
                images.sort()

                unique_images = []

                for image in images:
                    if image not in unique_images:
                        unique_images.append(image)

                if len(unique_images) != IMAGES_PER_TWIN:
                    print(
                        f"Skipping {twin_folder}. "
                        f"Expected {IMAGES_PER_TWIN} images but found {len(unique_images)}"
                    )
                    continue

                twin_data = {
                    "fold_id": fold_dir.name,
                    "twin_pair": twin_pair_folder.name,
                    "twin_name": twin_folder.name,
                    "images": unique_images
                }

                all_twins.append(twin_data)

    print(f"Individual twins found: {len(all_twins)}")

    return all_twins


def get_complete_twin_pairs(all_twins):
    twins_by_pair = {}

    for twin in all_twins:
        key = (twin["fold_id"], twin["twin_pair"])

        if key not in twins_by_pair:
            twins_by_pair[key] = []

        twins_by_pair[key].append(twin)

    complete_twin_pairs = {}

    for key, twins in twins_by_pair.items():
        if len(twins) == 2:
            complete_twin_pairs[key] = twins

    print(f"Complete twin pairs: {len(complete_twin_pairs)}")

    return complete_twin_pairs


def organize_pairs_by_fold(complete_twin_pairs):
    twin_pairs_by_fold = {}

    for key, twins in complete_twin_pairs.items():
        fold_id = key[0]
        twin_pair = key[1]

        if fold_id not in twin_pairs_by_fold:
            twin_pairs_by_fold[fold_id] = []

        twin_pairs_by_fold[fold_id].append((twin_pair, twins))

    fold_ids = list(twin_pairs_by_fold.keys())
    fold_ids.sort()

    print()
    print("Twin pairs per fold")

    for fold_id in fold_ids:
        count = len(twin_pairs_by_fold[fold_id])

        if count >= TWINS_PER_FOLD:
            status = "enough"
        else:
            status = "not enough"

        print(f"{fold_id}: {count} pairs ({status})")

    return twin_pairs_by_fold, fold_ids


def create_positive_pairs(twin):
    positive_pairs = []
    images = twin["images"]

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image_1 = images[i]
            image_2 = images[j]
            twin_name = twin["twin_name"]

            pair = (image_1, image_2, twin_name)
            positive_pairs.append(pair)

    return positive_pairs


def create_negative_pairs(twin_a, twin_b):
    negative_pairs = []

    for image_a in twin_a["images"]:
        for image_b in twin_b["images"]:
            pair = (image_a, image_b)
            negative_pairs.append(pair)

    return negative_pairs


def copy_positive_pairs(
    all_positive,
    gen_folder,
    gen_idx,
    fold_num,
    fold_id,
    twin_pair_name,
    positive_metadata
):
    for positive_pair in all_positive:
        img1 = positive_pair[0]
        img2 = positive_pair[1]
        twin_name = positive_pair[2]

        folder_name = str(gen_idx).zfill(4)
        pair_folder = gen_folder / folder_name
        pair_folder.mkdir(exist_ok=True)

        first_output = pair_folder / f"{gen_idx}_0.jpg"
        second_output = pair_folder / f"{gen_idx}_1.jpg"

        shutil.copy2(img1, first_output)
        shutil.copy2(img2, second_output)

        pair_info = {
            "pair_idx": gen_idx,
            "type": "genuine",
            "fold": fold_num,
            "twin_pair": twin_pair_name,
            "twin_name": twin_name,
            "fold_id": fold_id,
            "source1": str(img1),
            "source2": str(img2)
        }

        positive_metadata.append(pair_info)
        gen_idx += 1

    return gen_idx


def copy_negative_pairs(
    selected_negative,
    imp_folder,
    imp_idx,
    fold_num,
    fold_id,
    twin_pair_name,
    twin_a,
    twin_b,
    negative_metadata
):
    for negative_pair in selected_negative:
        img_a = negative_pair[0]
        img_b = negative_pair[1]

        folder_name = str(imp_idx).zfill(4)
        pair_folder = imp_folder / folder_name
        pair_folder.mkdir(exist_ok=True)

        first_output = pair_folder / f"{imp_idx}_0.jpg"
        second_output = pair_folder / f"{imp_idx}_1.jpg"

        shutil.copy2(img_a, first_output)
        shutil.copy2(img_b, second_output)

        pair_info = {
            "pair_idx": imp_idx,
            "type": "impostor",
            "fold": fold_num,
            "twin_pair": twin_pair_name,
            "twin_a_name": twin_a["twin_name"],
            "twin_b_name": twin_b["twin_name"],
            "fold_id": fold_id,
            "source1": str(img_a),
            "source2": str(img_b)
        }

        negative_metadata.append(pair_info)
        imp_idx += 1

    return imp_idx


def save_metadata(output_path, positive_metadata, negative_metadata):
    total_twin_pairs = NUM_FOLDS * TWINS_PER_FOLD
    total_pairs = total_twin_pairs * PAIRS_PER_TWIN_PAIR

    structure = {
        "num_folds": NUM_FOLDS,
        "twins_per_fold": TWINS_PER_FOLD,
        "images_per_twin": IMAGES_PER_TWIN,
        "positive_per_twin": POSITIVE_PER_TWIN,
        "positive_per_twin_pair": POSITIVE_PER_PAIR,
        "negative_per_twin_pair": NEGATIVE_PER_PAIR,
        "total_per_twin_pair": PAIRS_PER_TWIN_PAIR,
        "total_twin_pairs": total_twin_pairs,
        "total_pairs": total_pairs,
        "source_paths_added": True
    }

    metadata = {
        "positive": positive_metadata,
        "negative": negative_metadata,
        "structure": structure
    }

    metadata_path = output_path / "pair_metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    return metadata_path, total_twin_pairs, total_pairs


def create_balanced_pairs_per_twin(celeb_twins_folds, output_folder):
    celeb_twins_path = Path(celeb_twins_folds)
    output_path = Path(output_folder)

    gen_folder = output_path / "gen"
    imp_folder = output_path / "imp"

    gen_folder.mkdir(parents=True, exist_ok=True)
    imp_folder.mkdir(parents=True, exist_ok=True)

    all_twins = collect_twins(celeb_twins_path)
    complete_twin_pairs = get_complete_twin_pairs(all_twins)

    twin_pairs_by_fold, fold_ids = organize_pairs_by_fold(
        complete_twin_pairs
    )

    random.seed(42)

    positive_metadata = []
    negative_metadata = []

    gen_idx = 0
    imp_idx = 0

    for fold_num, fold_id in enumerate(fold_ids):
        print()
        print(f"Processing {fold_id}")

        selected_pairs = twin_pairs_by_fold[fold_id][:TWINS_PER_FOLD]

        if len(selected_pairs) < TWINS_PER_FOLD:
            raise ValueError(
                f"{fold_id} has {len(selected_pairs)} valid pairs. "
                f"{TWINS_PER_FOLD} are required."
            )

        for twin_pair_name, twins in selected_pairs:
            twin_a = twins[0]
            twin_b = twins[1]

            print(f"\nTwin pair: {twin_pair_name}")
            print(f"Twin A: {twin_a['twin_name']}, {len(twin_a['images'])} images")
            print(f"Twin B: {twin_b['twin_name']}, {len(twin_b['images'])} images")

            twin_a_positive = create_positive_pairs(twin_a)
            twin_b_positive = create_positive_pairs(twin_b)

            all_positive = twin_a_positive + twin_b_positive

            if len(all_positive) != POSITIVE_PER_PAIR:
                raise ValueError(
                    f"Expected {POSITIVE_PER_PAIR} positive pairs, "
                    f"but found {len(all_positive)}"
                )

            all_negative = create_negative_pairs(twin_a, twin_b)

            selected_negative = random.sample(
                all_negative,
                NEGATIVE_PER_PAIR
            )

            print(f"Positive pairs: {len(all_positive)}")
            print(
                f"Negative pairs: {len(selected_negative)} "
                f"out of {len(all_negative)}"
            )

            gen_idx = copy_positive_pairs(
                all_positive,
                gen_folder,
                gen_idx,
                fold_num,
                fold_id,
                twin_pair_name,
                positive_metadata
            )

            imp_idx = copy_negative_pairs(
                selected_negative,
                imp_folder,
                imp_idx,
                fold_num,
                fold_id,
                twin_pair_name,
                twin_a,
                twin_b,
                negative_metadata
            )

    metadata_path, total_twin_pairs, total_pairs = save_metadata(
        output_path,
        positive_metadata,
        negative_metadata
    )

    positive_count = len(positive_metadata)
    negative_count = len(negative_metadata)
    pair_count = positive_count + negative_count

    print()
    print(f"Twin pairs available: {len(complete_twin_pairs)}")
    print(f"Twin pairs used: {total_twin_pairs}")
    print(f"Positive pairs: {positive_count}")
    print(f"Negative pairs: {negative_count}")
    print(f"Total pairs: {pair_count}")
    print(f"Pairs per twin pair: {PAIRS_PER_TWIN_PAIR}")
    print(f"Expected total: {total_pairs}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate balanced twin pairs"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        default="Celeb_twins_pairs"
    )

    args = parser.parse_args()

    create_balanced_pairs_per_twin(
        celeb_twins_folds=args.name,
        output_folder=args.output
    )