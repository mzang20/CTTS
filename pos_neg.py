import shutil
from pathlib import Path
import random
import json

TWINS_PER_FOLD = 5
IMAGES_PER_TWIN = 12
POSITIVE_PER_TWIN = 66   # C(12, 2)
POSITIVE_PER_PAIR = 132  # 66 * 2
NEGATIVE_PER_PAIR = 132  # sampled from 144 possible (12 * 12)
PAIRS_PER_TWIN_PAIR = 264  # 132 + 132
NUM_FOLDS = 10

# Create pairs with equal distribution per twin pair
# 50 twins x 264 = 13200 total pairs
def create_balanced_pairs_per_twin(celeb_twins_folds, output_folder):
    celeb_twins_path = Path(celeb_twins_folds)
    output_path = Path(output_folder)

    gen_folder = output_path / "gen"
    imp_folder = output_path / "imp"
    gen_folder.mkdir(parents=True, exist_ok=True)
    imp_folder.mkdir(parents=True, exist_ok=True)

    print("Collecting all twins")
    all_twins = []

    for fold_dir in sorted(celeb_twins_path.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith('fold_'):
            continue

        for twin_pair_folder in fold_dir.iterdir():
            if not twin_pair_folder.is_dir():
                continue

            twin_folders = sorted([d for d in twin_pair_folder.iterdir() if d.is_dir()])

            for twin_folder in twin_folders:
                images = sorted(twin_folder.glob('*.jpg'))

                images = list(dict.fromkeys(images))

                if len(images) != IMAGES_PER_TWIN:
                    print(f"Skipping {twin_folder}: expected {IMAGES_PER_TWIN} images, found {len(images)}")
                    continue

                all_twins.append({
                    'fold_id': fold_dir.name,
                    'twin_pair': twin_pair_folder.name,
                    'twin_name': twin_folder.name,
                    'images': images
                })

    print(f"Total individual twins (with exactly {IMAGES_PER_TWIN} images): {len(all_twins)}")

    # Group twins by twin pair
    twins_by_pair = {}
    for twin in all_twins:
        key = (twin['fold_id'], twin['twin_pair'])
        if key not in twins_by_pair:
            twins_by_pair[key] = []
        twins_by_pair[key].append(twin)

    # Filter to only complete twin pairs (both twins present)
    complete_twin_pairs = {k: v for k, v in twins_by_pair.items() if len(v) == 2}
    print(f"Complete twin pairs: {len(complete_twin_pairs)}")

    # Organize twin pairs by fold
    twin_pairs_by_fold = {}
    for (fold_id, twin_pair), twins in complete_twin_pairs.items():
        if fold_id not in twin_pairs_by_fold:
            twin_pairs_by_fold[fold_id] = []
        twin_pairs_by_fold[fold_id].append((twin_pair, twins))

    print(f"\nTwin pairs per fold (before enforcing limit of {TWINS_PER_FOLD}):")
    for fold_id in sorted(twin_pairs_by_fold.keys()):
        count = len(twin_pairs_by_fold[fold_id])
        flag = "✓" if count >= TWINS_PER_FOLD else "✗ Insufficient"
        print(f"  {fold_id}: {count} pairs {flag}")

    # Generate pairs
    random.seed(42)

    positive_metadata = []
    negative_metadata = []

    gen_idx = 0
    imp_idx = 0

    for fold_num, fold_id in enumerate(sorted(twin_pairs_by_fold.keys())):
        print(f"\n{'='*60}")
        print(f"Processing {fold_id} (Fold {fold_num})...")
        print(f"{'='*60}")

        # Enforce exactly TWINS_PER_FOLD pairs per fold
        selected_pairs = twin_pairs_by_fold[fold_id][:TWINS_PER_FOLD]

        if len(selected_pairs) < TWINS_PER_FOLD:
            raise ValueError(
                f"{fold_id} only has {len(selected_pairs)} valid pairs, need {TWINS_PER_FOLD}. "
                f"Fix your dataset before proceeding."
            )

        for twin_pair_name, twins in selected_pairs:
            twin_a, twin_b = twins[0], twins[1]

            print(f"\n  Twin pair: {twin_pair_name}")
            print(f"    Twin A ({twin_a['twin_name']}): {len(twin_a['images'])} images")
            print(f"    Twin B ({twin_b['twin_name']}): {len(twin_b['images'])} images")

            # Generate ALL positive pairs for Twin A (66)
            twin_a_positive = []
            for i in range(len(twin_a['images'])):
                for j in range(i + 1, len(twin_a['images'])):
                    twin_a_positive.append((twin_a['images'][i], twin_a['images'][j], twin_a['twin_name']))

            # Generate ALL positive pairs for Twin B (66)
            twin_b_positive = []
            for i in range(len(twin_b['images'])):
                for j in range(i + 1, len(twin_b['images'])):
                    twin_b_positive.append((twin_b['images'][i], twin_b['images'][j], twin_b['twin_name']))

            all_positive = twin_a_positive + twin_b_positive
            assert len(all_positive) == POSITIVE_PER_PAIR, \
                f"Expected {POSITIVE_PER_PAIR} positive pairs, got {len(all_positive)}"

            # Generate ALL negative pairs (144 but sample 132)
            all_negative = [(img_a, img_b)
                            for img_a in twin_a['images']
                            for img_b in twin_b['images']]
            selected_negative = random.sample(all_negative, NEGATIVE_PER_PAIR)

            print(f"    Positive pairs: {len(all_positive)}")
            print(f"    Negative pairs: {len(selected_negative)} (sampled from {len(all_negative)})")

            # Copy positive pairs
            for img1, img2, twin_name in all_positive:
                pair_folder = gen_folder / str(gen_idx).zfill(4)
                pair_folder.mkdir(exist_ok=True)
                shutil.copy2(img1, pair_folder / f"{gen_idx}_0.jpg")
                shutil.copy2(img2, pair_folder / f"{gen_idx}_1.jpg")

                positive_metadata.append({
                    'pair_idx': gen_idx,
                    'type': 'genuine',
                    'fold': fold_num,
                    'twin_pair': twin_pair_name,
                    'twin_name': twin_name,
                    'fold_id': fold_id
                })

                gen_idx += 1

            # Copy negative pairs
            for img_a, img_b in selected_negative:
                pair_folder = imp_folder / str(imp_idx).zfill(4)
                pair_folder.mkdir(exist_ok=True)
                shutil.copy2(img_a, pair_folder / f"{imp_idx}_0.jpg")
                shutil.copy2(img_b, pair_folder / f"{imp_idx}_1.jpg")

                negative_metadata.append({
                    'pair_idx': imp_idx,
                    'type': 'impostor',
                    'fold': fold_num,
                    'twin_pair': twin_pair_name,
                    'twin_a_name': twin_a['twin_name'],
                    'twin_b_name': twin_b['twin_name'],
                    'fold_id': fold_id
                })

                imp_idx += 1

    # Save metadata
    metadata = {
        'positive': positive_metadata,
        'negative': negative_metadata,
        'structure': {
            'num_folds': NUM_FOLDS,
            'twins_per_fold': TWINS_PER_FOLD,
            'images_per_twin': IMAGES_PER_TWIN,
            'positive_per_twin': POSITIVE_PER_TWIN,
            'positive_per_twin_pair': POSITIVE_PER_PAIR,
            'negative_per_twin_pair': NEGATIVE_PER_PAIR,
            'total_per_twin_pair': PAIRS_PER_TWIN_PAIR,
            'total_twin_pairs': NUM_FOLDS * TWINS_PER_FOLD,
            'total_pairs': NUM_FOLDS * TWINS_PER_FOLD * PAIRS_PER_TWIN_PAIR
        }
    }

    with open(output_path / "pair_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print(f"Total twin pairs available: {len(complete_twin_pairs)}, used: {NUM_FOLDS * TWINS_PER_FOLD}")
    print(f"Total positive pairs:       {len(positive_metadata)}")
    print(f"Total negative pairs:       {len(negative_metadata)}")
    print(f"Total pairs:                {len(positive_metadata) + len(negative_metadata)}")
    print(f"\nPer-twin-pair breakdown:")
    print(f"  Images per twin:          {IMAGES_PER_TWIN}")
    print(f"  Positive pairs:           {POSITIVE_PER_PAIR}")
    print(f"  Negative pairs:           {NEGATIVE_PER_PAIR}")
    print(f"  Total per twin pair:      {PAIRS_PER_TWIN_PAIR}")
    print(f"  Expected grand total:     {NUM_FOLDS * TWINS_PER_FOLD * PAIRS_PER_TWIN_PAIR}")
    print("="*60)


if __name__ == "__main__":
    create_balanced_pairs_per_twin(
        celeb_twins_folds="Celeb_twins_folds",
        output_folder="Celeb_twins_pairs"
    )