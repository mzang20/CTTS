import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import DataParallel

import verification
from data.load_test_sets_recognition import get_val_pair
from model import get_vit, iresnet

NUM_FOLDS = 10
TWINS_PER_FOLD = 8
POSITIVE_PER_PAIR = 132
NEGATIVE_PER_PAIR = 132

GEN_PER_FOLD = TWINS_PER_FOLD * POSITIVE_PER_PAIR   # 1056
IMP_PER_FOLD = TWINS_PER_FOLD * NEGATIVE_PER_PAIR   # 1056
PAIRS_PER_FOLD = GEN_PER_FOLD + IMP_PER_FOLD        # 2112


class Test:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.args = args
        self.model = self.create_model(args)
        self.model = self.model.to(self.device)

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Model will use {gpu_count} GPUs!")
            self.model = DataParallel(self.model)
        elif gpu_count == 1:
            print("Model will use 1 GPU.")
        self.validation_list = []
        for val_name in args.val_list:
            dataset, issame = get_val_pair(args.val_source, val_name)
            validation_data = [dataset, issame, val_name]
            self.validation_list.append(validation_data)

    def create_model(self, args):
        if args.model == "iresnet":
            model = iresnet(args.depth, fp16=True)
        elif args.model == "vit":
            model = get_vit(args.depth)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        model.load_state_dict(torch.load(args.model_path))
        return model

    def evaluate(self):
        self.model.eval()

        if not self.validation_list:
            raise ValueError("No validation datasets were provided.")

        validation_results = []

        print("Validating...")

        for validation in self.validation_list:
            dataset = validation[0]
            issame = validation[1]
            val_name = validation[2]

            mean_accuracy, standard_deviation = self.evaluate_recognition(
                dataset,
                issame
            )

            result = {
                "dataset": val_name,
                "mean_accuracy": mean_accuracy,
                "standard_deviation": standard_deviation
            }

            validation_results.append(result)

            analysis_requested = (
                self.args.plot_twin is not None
                or self.args.plot_all_twins
                or self.args.find_outliers
            )

            if val_name == "twins" and analysis_requested:
                print()
                print("Analyzing twin pairs")
                self.analyze_twin_pairs(dataset, issame)

        print()
        print("Validation results")

        for result in validation_results:
            print(
                f"{result['dataset']}: "
                f"{result['mean_accuracy']:.5f}+-"
                f"{result['standard_deviation']:.5f}"
            )

    def l2_norm(self, tensor: torch.Tensor, axis=1):
        norm = torch.norm(tensor, 2, axis, True)
        normalized_tensor = torch.div(tensor, norm)
        return normalized_tensor, norm

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        normalized_embeddings = self.compute_embeddings(samples)

        _, _, accuracy = verification.evaluate(
            normalized_embeddings,
            issame,
            nrof_folds,
            cosine=self.args.cosine
        )

        mean_accuracy = round(accuracy.mean(), 5)
        standard_deviation = round(accuracy.std(), 5)
        return mean_accuracy, standard_deviation

    # Genuine and impostor pair indices restart independently in each source folder.
    def pair_idx_to_position(self, pair_idx, fold, is_positive):
        fold_start = fold * PAIRS_PER_FOLD

        if is_positive:
            offset = pair_idx - fold * GEN_PER_FOLD
            position = fold_start + offset
        else:
            offset = pair_idx - fold * IMP_PER_FOLD
            position = fold_start + GEN_PER_FOLD + offset

        return position

    def pair_images(self, rec):
        source1 = rec.get('source1')
        source2 = rec.get('source2')

        if source1 and source2:
            images_original = True
        else:
            source1 = None
            source2 = None
            images_original = False

        return source1, source2, images_original

    def compute_embeddings(self, samples):
        num_images = len(samples) // 2
        embeddings = np.zeros([num_images, 512])

        if self.args.add_flip or self.args.add_norm:
            use_flip = True
        else:
            use_flip = False

        with torch.no_grad():
            for idx in range(0, num_images, self.args.batch_size):
                batch_end = idx + self.args.batch_size
                batch_samples = samples[idx:batch_end]
                batch_or = torch.tensor(batch_samples)
                batch_size = batch_or.shape[0]
                batch_or = batch_or.to(self.device)

                if use_flip:
                    flip_start = num_images + idx
                    flip_end = flip_start + batch_size
                    flip_samples = samples[flip_start:flip_end]
                    batch_flip = torch.tensor(flip_samples)
                    batch_flip = batch_flip.to(self.device)

                if self.args.add_flip:
                    original_output = self.model(batch_or).cpu()
                    flipped_output = self.model(batch_flip).cpu()
                    combined_output = original_output + flipped_output
                    embeddings[idx:idx + batch_size] = combined_output
                elif self.args.add_norm:
                    original_output = self.model(batch_or)
                    flipped_output = self.model(batch_flip)

                    emb_or, norm_or = self.l2_norm(original_output, axis=1)
                    emb_flip, norm_flip = self.l2_norm(flipped_output, axis=1)

                    combined_output = emb_or * norm_or + emb_flip * norm_flip
                    embeddings[idx:idx + batch_size] = combined_output.cpu()
                else:
                    original_output = self.model(batch_or).cpu()
                    embeddings[idx:idx + batch_size] = original_output

        embedding_norms = np.linalg.norm(embeddings, 2, 1, True)
        normalized_embeddings = np.divide(embeddings, embedding_norms)
        return normalized_embeddings

    def compute_distances(self, normalized_embeddings):
        embeddings1 = normalized_embeddings[0::2]
        embeddings2 = normalized_embeddings[1::2]

        difference = embeddings1 - embeddings2
        squared_difference = np.square(difference)
        distances = np.sum(squared_difference, axis=1)
        return distances

    def compute_fold_thresholds(self, dist, issame_array):
        from sklearn.model_selection import KFold

        thresholds = np.arange(0, 4, 0.01)

        k_fold = KFold(n_splits=NUM_FOLDS, shuffle=False)
        fold_thresholds = {}

        for fold_idx, (train_set, _) in enumerate(k_fold.split(dist)):
            acc_train = np.zeros(len(thresholds))
            for t_idx, threshold in enumerate(thresholds):
                predict_issame = np.less(dist[train_set], threshold)
                tp = np.sum(np.logical_and(predict_issame, issame_array[train_set]))
                tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                                           np.logical_not(issame_array[train_set])))
                acc_train[t_idx] = float(tp + tn) / len(train_set)

            best_idx = np.argmax(acc_train)
            fold_thresholds[fold_idx] = thresholds[best_idx]
            print(f"Fold {fold_idx} threshold: {thresholds[best_idx]:.6f}, "
                  f"train accuracy: {acc_train[best_idx]:.4f}")

        return fold_thresholds

    def build_twin_pair_data(self, metadata):
        twin_pair_data = {}

        for pos in metadata['positive']:
            key = (pos['fold'], pos['twin_pair'])

            if key not in twin_pair_data:
                twin_pair_data[key] = {
                    'pair_positions': [],
                    'pair_records': []
                }

            pair_data = twin_pair_data[key]
            position = self.pair_idx_to_position(
                pos['pair_idx'],
                pos['fold'],
                is_positive=True
            )

            positive_record = {
                'position': position,
                'pair_idx': pos['pair_idx'],
                'is_positive': True,
                'twin_name': pos.get('twin_name'),
                'source1': pos.get('source1'),
                'source2': pos.get('source2')
            }

            pair_data['pair_positions'].append(position)
            pair_data['pair_records'].append(positive_record)

        for neg in metadata['negative']:
            key = (neg['fold'], neg['twin_pair'])

            if key not in twin_pair_data:
                twin_pair_data[key] = {
                    'pair_positions': [],
                    'pair_records': []
                }

            pair_data = twin_pair_data[key]
            position = self.pair_idx_to_position(
                neg['pair_idx'],
                neg['fold'],
                is_positive=False
            )

            negative_record = {
                'position': position,
                'pair_idx': neg['pair_idx'],
                'is_positive': False,
                'twin_a_name': neg.get('twin_a_name'),
                'twin_b_name': neg.get('twin_b_name'),
                'source1': neg.get('source1'),
                'source2': neg.get('source2')
            }

            pair_data['pair_positions'].append(position)
            pair_data['pair_records'].append(negative_record)

        return twin_pair_data

    def plot_twin_similarity_histogram(self, dist, issame_array, twin_pair_data, fold_thresholds, twin_pair_name):
        target_key = None
        search_name = twin_pair_name.lower()

        for key in twin_pair_data:
            current_twin_pair = key[1].lower()
            if search_name in current_twin_pair:
                target_key = key
                break

        if target_key is None:
            print(f"Error: Twin pair '{twin_pair_name}' not found")
            return

        fold, twin_pair = target_key
        data = twin_pair_data[target_key]
        threshold = fold_thresholds[fold]

        pair_positions = data['pair_positions']
        pair_dists = []
        pair_labels = []

        for position in pair_positions:
            pair_dists.append(dist[position])
            pair_labels.append(issame_array[position])

        positive_dists = []
        negative_dists = []

        for index in range(len(pair_dists)):
            distance = pair_dists[index]
            label = pair_labels[index]

            if label:
                positive_dists.append(distance)
            else:
                negative_dists.append(distance)

        print()
        print(f"Distance statistics for {twin_pair}")
        print(f"Fold: {fold}, threshold: {threshold:.6f}")
        def stats(values):
            values = np.asarray(values, dtype=float)

            if values.size == 0:
                result = "count=0"
            else:
                count = values.size
                minimum = values.min()
                mean = values.mean()
                median = np.median(values)
                maximum = values.max()
                standard_deviation = values.std()

                result = (
                    f"count={count}, min={minimum:.6f}, mean={mean:.6f}, "
                    f"median={median:.6f}, max={maximum:.6f}, "
                    f"std={standard_deviation:.6f}"
                )

            return result
        print(f"Positive pairs: {stats(positive_dists)}")
        print(f"Negative pairs: {stats(negative_dists)}")

        all_dists = positive_dists + negative_dists

        # Use a fixed bin width; optionally share the same x-axis limit across plots.
        bin_width = self.args.hist_bin_width
        if self.args.hist_xmax is not None:
            xmax = self.args.hist_xmax
            n_over = 0
            for distance in all_dists:
                if distance > xmax:
                    n_over = n_over + 1

            if n_over > 0:
                print(f"Warning: {n_over} distances exceed --hist_xmax={xmax:g} and will not be shown")
        else:
            xmax = max(np.max(all_dists), threshold) + 5 * bin_width
        bins = np.arange(0, xmax + bin_width, bin_width)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(positive_dists, bins=bins, alpha=0.6, color='green',
                label=f'Positive pairs (same twin, n={len(positive_dists)})', edgecolor='black')
        ax.hist(negative_dists, bins=bins, alpha=0.6, color='red',
                label=f'Negative pairs (different twins, n={len(negative_dists)})', edgecolor='black')
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
        ax.set_xlim(0, xmax)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distance Distribution for {twin_pair}\n(Fold {fold})', fontsize=14)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

        pos_correct = 0
        for distance in positive_dists:
            if distance < threshold:
                pos_correct = pos_correct + 1

        neg_correct = 0
        for distance in negative_dists:
            if distance >= threshold:
                neg_correct = neg_correct + 1
        accuracy = (pos_correct + neg_correct) / len(all_dists)
        textstr = (f'Accuracy: {accuracy:.2%}\n'
                   f'Positive correct: {pos_correct}/{len(positive_dists)} ({pos_correct/len(positive_dists):.1%})\n'
                   f'Negative correct: {neg_correct}/{len(negative_dists)} ({neg_correct/len(negative_dists):.1%})')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()

        hist_dir = Path(self.args.metadata_path).parent.parent / "histograms"
        hist_dir.mkdir(exist_ok=True)
        output_file = hist_dir / f"{twin_pair}_distance_histogram.png"

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {output_file}")
        plt.close(fig)

    @staticmethod
    def outlier_reason(bucket, thr, misclassified):
        if bucket == 'above':
            reason = f"dist>={thr:g}"
        else:
            reason = f"dist<={thr:g}"

        if misclassified:
            reason = reason + ",MISCLASSIFIED"

        return reason

    def make_entry(self, r, d, twin_pair, fold, fold_thresholds, bucket=None, thr_val=None):
        img1, img2, is_orig = self.pair_images(r)
        threshold = float(fold_thresholds[fold])

        if r['is_positive']:
            label = 'same_person'
            misclassified = bool(d >= threshold)
        else:
            label = 'different_twins'
            misclassified = bool(d < threshold)

        entry = {
            'twin_pair': twin_pair,
            'fold': fold,
            'label': label,
            'pair_idx': r['pair_idx'],
            'distance': d,
            'threshold': threshold,
            'misclassified': misclassified,
            'image1': img1,
            'image2': img2,
            'images_original': is_orig
        }

        if r['is_positive']:
            entry['twin_name'] = r.get('twin_name')
        else:
            entry['twin_a_name'] = r.get('twin_a_name')
            entry['twin_b_name'] = r.get('twin_b_name')

        if bucket is not None:
            entry['bucket'] = bucket
            entry['reason'] = self.outlier_reason(
                bucket,
                thr_val,
                misclassified
            )

        return entry

    # Collect pairs outside the requested distance range, grouped by twin pair.
    def find_outlier_pairs(self, dist, issame_array, twin_pair_data, fold_thresholds):
        dist_low = self.args.dist_low
        dist_high = self.args.dist_high

        dist = np.asarray(dist)
        issame_array = np.asarray(issame_array)
        n_pairs = issame_array.shape[0]
        if dist.ndim != 1 or dist.shape[0] != n_pairs:
            raise ValueError(
                f"Expected one distance per pair: dist shape={dist.shape}, "
                f"labels={n_pairs}"
            )

        if self.args.outlier_output:
            txt_path = Path(self.args.outlier_output)
        else:
            txt_path = Path(self.args.metadata_path).parent / "outlier_pairs.txt"
        json_path = txt_path.with_suffix(".json")

        per_twin_output = []

        labels = issame_array.astype(bool, copy=False)
        neg_all = dist[~labels]
        pos_all = dist[labels]
        print()
        print("Distance ranges by label")
        print(f"Negative pairs: n={len(neg_all)}, min={neg_all.min():.4f}, "
              f"median={np.median(neg_all):.4f}, max={neg_all.max():.4f}")
        print(f"Positive pairs: n={len(pos_all)}, min={pos_all.min():.4f}, "
              f"median={np.median(pos_all):.4f}, max={pos_all.max():.4f}")
        n_below = int((dist <= dist_low).sum())
        n_above = int((dist >= dist_high).sum())
        print(f"Pairs with distance <= {dist_low:g}: {n_below} "
              f"(positive {int((pos_all <= dist_low).sum())}, negative {int((neg_all <= dist_low).sum())})")
        print(f"Pairs with distance >= {dist_high:g}: {n_above} "
              f"(positive {int((pos_all >= dist_high).sum())}, negative {int((neg_all >= dist_high).sum())})")

        for (fold, twin_pair), data in sorted(twin_pair_data.items(),
                                              key=lambda kv: kv[0][1].lower()):
            threshold = float(fold_thresholds[fold])
            records = data['pair_records']

            below = []
            above = []
            for r in records:
                d = float(dist[r['position']])
                if d <= dist_low:
                    entry = self.make_entry(
                        r,
                        d,
                        twin_pair,
                        fold,
                        fold_thresholds,
                        'below',
                        dist_low
                    )
                    below.append(entry)
                elif d >= dist_high:
                    entry = self.make_entry(
                        r,
                        d,
                        twin_pair,
                        fold,
                        fold_thresholds,
                        'above',
                        dist_high
                    )
                    above.append(entry)

            if not below and not above:
                continue

            below.sort(key=lambda e: e['distance'])                # closest first
            above.sort(key=lambda e: e['distance'], reverse=True)  # farthest first

            per_twin_output.append({
                'twin_pair': twin_pair, 'fold': fold, 'threshold': threshold,
                'num_below': len(below), 'num_above': len(above),
                'below': below, 'above': above,
            })

        with open(json_path, 'w') as f:
            json.dump({
                'params': {'dist_low': dist_low, 'dist_high': dist_high},
                'per_twin': per_twin_output,
            }, f, indent=2)

        def format_entry(e):
            if e['label'] == 'same_person':
                label_tag = "same-person"
            else:
                label_tag = "different-twins"

            if e['misclassified']:
                marker = " misclassified"
            else:
                marker = ""

            result = (
                f"distance={e['distance']:.4f}, threshold={e['threshold']:.4f}, "
                f"label={label_tag}, pair={e['pair_idx']}, reason={e['reason']}{marker}\n"
                f"img1={e['image1']}\n"
                f"img2={e['image2']}\n"
            )
            return result

        with open(txt_path, 'w') as f:
            f.write("Extreme image pairs\n")
            f.write("Metric: squared Euclidean distance on normalized embeddings\n")
            f.write(f"Low cutoff: {dist_low:g}\n")
            f.write(f"High cutoff: {dist_high:g}\n\n")

            if not per_twin_output:
                f.write("No pairs found.\n")

            for t in per_twin_output:
                f.write(f"{t['twin_pair']}\n")
                f.write(f"fold={t['fold']}, threshold={t['threshold']:.4f}\n")
                f.write(f"below={t['num_below']}, above={t['num_above']}\n")

                if t['below']:
                    f.write(f"below {dist_low:g}\n")
                    for e in t['below']:
                        f.write(format_entry(e))

                if t['above']:
                    f.write(f"above {dist_high:g}\n")
                    for e in t['above']:
                        f.write(format_entry(e))

                f.write("\n")

        print()
        print("Outlier detection complete")
        print(f"Distance range: <= {dist_low:g} or >= {dist_high:g}")
        print(f"Twin pairs found: {len(per_twin_output)}")
        print(f"TXT: {txt_path}")
        print(f"JSON: {json_path}")

    def analyze_twin_pairs(self, samples, issame):
        if not self.args.metadata_path:
            print("Error: --metadata_path required for twin pair analysis")
            return

        metadata_path = Path(self.args.metadata_path)
        if not metadata_path.exists():
            print(f"Error: Metadata file not found at {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # The loader stores two original images per pair, followed by their flips.
        expected_samples = 4 * len(issame)
        if len(samples) != expected_samples:
            print(
                f"WARNING: expected {expected_samples} images, "
                f"found {len(samples)}. Rebuild the test set."
            )

        print("Computing embeddings")
        normalized_embeddings = self.compute_embeddings(samples)

        expected_embeddings = 2 * len(issame)
        if normalized_embeddings.shape[0] != expected_embeddings:
            raise ValueError(
                f"Expected {expected_embeddings} embeddings, "
                f"got {normalized_embeddings.shape[0]}"
            )

        print("Computing distances")
        issame_array = np.array(issame)
        dist = self.compute_distances(normalized_embeddings)

        print("Building index mapping")
        twin_pair_data = self.build_twin_pair_data(metadata)

        print("Computing fold thresholds")
        fold_thresholds = self.compute_fold_thresholds(dist, issame_array)

        print("Computing twin pair accuracies")
        twin_pair_results = []

        for (fold, twin_pair), data in twin_pair_data.items():
            threshold = fold_thresholds[fold]
            pair_positions = data['pair_positions']
            pair_dists = []
            pair_labels = []

            for position in pair_positions:
                pair_dists.append(dist[position])
                pair_labels.append(issame_array[position])

            pos_correct = 0
            neg_correct = 0
            pos_total = 0
            neg_total = 0

            for index in range(len(pair_dists)):
                distance = pair_dists[index]
                label = pair_labels[index]

                if label:
                    pos_total = pos_total + 1
                    if distance < threshold:
                        pos_correct = pos_correct + 1
                else:
                    neg_total = neg_total + 1
                    if distance >= threshold:
                        neg_correct = neg_correct + 1

            overall_accuracy = (pos_correct + neg_correct) / len(pair_positions)

            if pos_total > 0:
                positive_accuracy = pos_correct / pos_total
            else:
                positive_accuracy = 0

            if neg_total > 0:
                negative_accuracy = neg_correct / neg_total
            else:
                negative_accuracy = 0

            result = {
                'twin_pair': twin_pair,
                'fold': fold,
                'threshold': threshold,
                'accuracy': overall_accuracy,
                'positive_correct': pos_correct,
                'positive_total': pos_total,
                'negative_correct': neg_correct,
                'negative_total': neg_total,
                'positive_accuracy': positive_accuracy,
                'negative_accuracy': negative_accuracy
            }
            twin_pair_results.append(result)

        twin_pair_results.sort(key=lambda x: x['accuracy'])

        print()
        print("Twin pairs ranked by difficulty")
        for i, result in enumerate(twin_pair_results, 1):
            print(f"\n{i}. {result['twin_pair']}")
            print(f"Overall accuracy: {result['accuracy']:.2%}")
            print(f"Fold: {result['fold']}, threshold: {result['threshold']:.6f}")
            print(f"Positive: {result['positive_correct']}/{result['positive_total']} ({result['positive_accuracy']:.2%})")
            print(f"Negative: {result['negative_correct']}/{result['negative_total']} ({result['negative_accuracy']:.2%})")

        if self.args.find_outliers:
            print("Finding outlier pairs")
            self.find_outlier_pairs(dist, issame_array, twin_pair_data, fold_thresholds)

        if self.args.plot_twin:
            print(f"Generating histogram for {self.args.plot_twin}")
            self.plot_twin_similarity_histogram(dist, issame_array, twin_pair_data,
                                                fold_thresholds, self.args.plot_twin)
        if self.args.plot_all_twins:
            print("Generating histograms for all twin pairs")
            for key in twin_pair_data.keys():
                twin_pair = key[1]
                print(twin_pair)
                self.plot_twin_similarity_histogram(
                    dist,
                    issame_array,
                    twin_pair_data,
                    fold_thresholds,
                    twin_pair
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a face-recognition model.")
    parser.add_argument("--model_path", "-model_path", help="Path to model weights.", type=str)
    parser.add_argument("--model", "-model", help="Model type: iresnet or vit.", type=str, default="iresnet")
    parser.add_argument("--depth", "-d",
        help="Model depth or ViT size.", default="100", type=str)
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--add_flip", "-aflip",
        help="Sum original and flipped-image features.",
        action="store_true")
    parser.add_argument("--add_norm", "-anorm", help="Add feature norm.", action="store_true")
    parser.add_argument("--cosine", "-cosine", help="Use cosine distance.", action="store_true")
    parser.add_argument("--val_list", "-v",
        help="List of datasets to validate.",
        nargs="+", default=["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw", "hadrian", "eclipse"])
    parser.add_argument("--val_source", "-vs",
        help="Path to the val images or dataset LMDB file.", default="./test_sets")
    parser.add_argument("--metadata_path", "-meta",
        help="Path to pair_metadata.json for twin pair analysis.", type=str, default=None)
    parser.add_argument("--plot_twin", "-pt",
        help="Generate similarity histogram for a specific twin pair (e.g., 'Ashmore_twins').",
        default=None)
    parser.add_argument("--plot_all_twins", "-pat",
        help="Generate histograms for all twin pairs.", action="store_true")
    parser.add_argument("--hist_xmax", "-hxm",
        help="Shared x-axis maximum for twin histograms.",
        type=float, default=None)
    parser.add_argument("--hist_bin_width", "-hbw",
        help="Histogram bin width.",
        type=float, default=0.05)

    parser.add_argument("--find_outliers", "-fo",
        help="Dump image pairs with extreme absolute distance to txt+json.",
        action="store_true")
    parser.add_argument("--dist_low", "-dl",
        help="Lower distance cutoff for outliers.",
        type=float, default=0.2)
    parser.add_argument("--dist_high", "-dh",
        help="Upper distance cutoff for outliers.",
        type=float, default=1.4)
    parser.add_argument("--outlier_output", "-oo",
        help="Output TXT path; JSON uses the same stem.",
        type=str, default=None)

    args = parser.parse_args()
    test = Test(args)
    test.evaluate()