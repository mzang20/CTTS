# CTTS: Celeb Twins Test Set for Benchmarking Face Recognition Accuracy

This repository provides tools to construct and evaluate the **Celeb Twins Test Set (CTTS)** — an in-the-wild twin verification benchmark for face recognition systems. CTTS extends the [Goldilocks Test Sets for Face Verification](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) framework by introducing identical (monozygotic) twin verification under unconstrained conditions.

## Overview

Existing twin verification benchmarks such as ND-Twins rely on controlled laboratory imagery. CTTS addresses this gap by using celebrity twin photos sourced from public appearances where pose, lighting, expression, and image quality vary naturally. This makes CTTS more representative of real-world deployment conditions for face recognition systems.

## Dataset Documentation

The complete list of twin pairs included in CTTS is provided in the [Twins_List](Twins_List.pdf) document.

For each of the 80 monozygotic twin sets, the document provides:

- Twin names and birth year
- A source verifying that the pair is monozygotic or identical
- Notes on visually distinguishing skin marks (**SM**) when readily apparent
- Notes on known left/right-handed or left/right-footed differences (**LR**), which may indicate mirror twins

## Citation

If you find this dataset or code helpful, please cite:

```bibtex
@article{CTTS2026,
  title={The Celeb Twins Test Set for Benchmarking Face Recognition Accuracy},
  author={Zang, Michael and Wu, Haiyu and Bowyer, Kevin W.},
  year={2026}
}
```

This work builds on the Goldilocks framework, please also cite:

```bibtex
@article{Goldilocks_CVPR_2026,
  title={Goldilocks Test Sets for Face Verification},
  author={Wu, Haiyu and Tian, Sicong and Bhatta, Aman and Gutierrez, Jacob and Bezold, Grace and Argueta, Genesis and Ricanek Jr., Karl and King, Michael C. and Bowyer, Kevin W.},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Pipeline](#pipeline)
  - [Step 1: Generate Pairs (`pos_neg.py`)](#step-1-generate-pairs-pos_negpy)
  - [Step 2: Build Test Set (`prepare_test_images.py`)](#step-2-build-test-set-prepare_test_imagespy)
  - [Step 3: Evaluate (`test1.py`)](#step-3-evaluate-test1py)
- [Supported Models](#supported-models)
- [Per-Twin-Pair Analysis](#per-twin-pair-analysis)
- [Acknowledgement](#acknowledgement)

## Dataset Preparation

You can request access to the dataset by completing this [form](https://docs.google.com/forms/d/e/1FAIpQLSea5sjmCRVt1UcTLP_YBeZDre94qOVHPMBH_5XyGXQnmyVvqg/viewform?usp=sharing&ouid=115789632013493782610). A Google account is required to access the dataset. Upon approval, access will be granted to the Google account provided in the form, and a link to the dataset will be sent to that email address. This email address will also be used for future dataset release notes and updates, including corrections, additions, or changes to the metadata. Once access has been granted, follow the instructions under [Input Structure](#input-structure) to prepare the dataset.

### Input Structure

Organize your dataset into 10 folds with twin pair subfolders, each containing two individual twin folders with 12 aligned images:

```
Celeb_twins_folds/
├── fold_00/
│   ├── Daniel_twins/
│   │   ├── Daniel_Brittany/
│   │   │   ├── Daniel_Brittany_01.jpg
│   │   │   └── ... (12 images)
│   │   └── Daniel_Cynthia/
│   │       └── ... (12 images)
│   ├── Sprouse_twins/
│   │   ├── Sprouse_Cole/
│   │   └── Sprouse_Dylan/
│   └── ... (5 pairs)
├── fold_01/
└── ... (10 folds)
```
Each fold holds **8 twin pairs** (10 folds × 8 = 80 pairs total), and every individual twin folder must contain exactly **12 aligned images**. Folds are twin-disjoint, so no twin pair appears in more than one fold.

Run `organize.py` to distribute the images into the layout above. Set `SRC` (source directory of aligned pairs) and `DST` (`Celeb_twins_folds`) at the top of the script. Run once with `DRY_RUN = True` to preview the moves then set `DRY_RUN = False` and rerun:

```bash
python organize.py
```

### Training Sets

CTTS is evaluated using pre-trained models. We support weights trained on:

- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
- [WebFace4M](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

## Pipeline

CTTS evaluation runs in three stages. Generate verification pairs, pack them into the standard test-set format, then evaluate a pre-trained model. All three scripts share the same constants (10 folds, 8 twin pairs per fold, 12 images per twin) so they must be kept consistent if modified.

### Step 1: Generate Pairs (`pos_neg.py`)
 
For each of the 80 twin pairs, this generates a balanced set of **264 pairs**:
 
- **132 genuine (same-person) pairs**: all 66 within-person combinations for each of the two twins (66 x 2).
- **132 impostor (different-twin) pairs**: sampled from the 144 possible cross-twin combinations (fixed seed 42).
  
Image pairs are copied into `gen/` (genuine) and `imp/` (impostor) subfolders, and a `pair_metadata.json` file records each pair's fold, twin identity, and source image paths.
 
```bash
python pos_neg.py --name Celeb_twins_folds --output Celeb_twins_pairs
```
 
**Output:** `Celeb_twins_pairs/{gen, imp, pair_metadata.json}`: 10 folds × 8 pairs × 264 = **21,120 pairs** (10,560 genuine + 10,560 impostor).

### Step 2: Build Test Set (`prepare_test_images.py`)
 
Reads the `gen/` then `imp/` folders in fold order, resizes each image to 112 × 112, converts BGR → RGB, and packs everything into a single `.npy` array with an `issame.txt` label file. Each fold is verified to contain exactly 1,056 genuine + 1,056 impostor = 2,112 pairs.
 
```bash
python prepare_test_images.py --pairs_folder Celeb_twins_pairs --destination output --dataset_name twins
```
 
**Output:** `./test_sets/twins/twins.npy` and `./test_sets/twins/issame.txt`.

### Step 3: Evaluate (`test.py`)
 
Loads the test set, extracts embeddings, computes squared-Euclidean distance on L2-normalized features, and runs 10-fold cross-validation to report mean accuracy ± standard deviation.
 
```bash
python test.py --model_path arcface-r100-ms1mv2.pth --model iresnet --depth 100 --val_source output --val_list twins --batch_size 128 --add_flip
```
 
Use the test flag that matches your model (see [Supported Models](#supported-models)). The dataset name must be `twins` for the per-twin analysis in the next section to activate.

## Supported Models

CTTS can be evaluated with any model compatible with the [SOTA-FR-train-and-test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) framework:

| Method | Conference | Test Flag |
|---|---|---|
| ArcFace | CVPR19 | `--add_flip` |
| MagFace | CVPR21 | `--add_flip` |
| AdaFace | CVPR22 | `--add_norm` |
| UniFace | ICCV23 | `--add_flip` |

## Per-Twin-Pair Analysis
 
`test1.py` extends `test.py` and can break results down per twin pair using fold-specific optimal thresholds. These options require `--metadata_path` pointing at the `pair_metadata.json` produced in Step 1.
 
**Rank twin pairs by difficulty**: (hardest first), with per-pair genuine/impostor accuracy and summary statistics:
 
```bash
python test1.py --model_path arcface-r100-ms1mv2.pth --model iresnet --depth 100 --val_source output --val_list twins --batch_size 128 --metadata_path Celeb_twins_pairs\pair_metadata.json --add_flip
```
 
**Distance histograms**: plot the genuine vs. impostor distance distribution for one twin pair (`--plot_twin`) or all of them (`--plot_all_twins`). Images are written to a `histograms/` directory:
 
```bash
# add to the command above:
--plot_twin Daniel_twins        # single pair
--plot_all_twins                # every pair
```
 
**Outlier pairs**: dump the most extreme image pairs by absolute distance, label-agnostically (very close impostors and very far genuine pairs), to `outlier_pairs.txt` and a `.json`. Tune the outliers with `--dist_low` / `--dist_high`:
 
```bash
# add to the first command:
--find_outliers --dist_low 0.2 --dist_high 1.4
```

## Acknowledgement

This work builds on the [SOTA-FR-train-and-test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) framework by Haiyu Wu et al. We thank the authors for their valuable contributions to the face recognition community.

## License

[MIT License](https://github.com/mzang20/CTTS/blob/main/license.md) for the code and model weights
