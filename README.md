# CTTS: Celeb Twins Test Set for Benchmarking Face Recognition Accuracy

This repository provides tools to construct and evaluate the **Celeb Twins Test Set (CTTS)** — an in-the-wild twin verification benchmark for face recognition systems. CTTS extends the [Goldilocks Test Sets for Face Verification](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) framework by introducing identical (monozygotic) twin verification under unconstrained conditions.

## Overview

Existing twin verification benchmarks such as ND-Twins rely on controlled laboratory imagery. CTTS addresses this gap by using celebrity twin photos sourced from public appearances where pose, lighting, expression, and image quality vary naturally. This makes CTTS more representative of real-world deployment conditions for face recognition systems.

## Citation

If you find this dataset or code helpful, please cite:

```bibtex
@article{CTTS2026,
  title={The Celeb Twins Test Set for Benchmarking Face Recognition Accuracy},
  author={TODO},
  year={2026}
}
```

This work builds on the Goldilocks framework — please also cite:

```bibtex
@article{GoldilocksFRTestSet2024,
  title={Goldilocks Test Sets for Face Verification},
  author={Wu, Haiyu and Tian, Sicong and Bhatta, Aman and Gutierrez, Jacob and Bezold, Grace and Argueta, Genesis and Ricanek Jr., Karl and King, Michael C. and Bowyer, Kevin W.},
  year={2024}
}
```

## Table of Contents

- [Environment](#environment)
- [Dataset Preparation](#dataset-preparation)
- [Pipeline](#pipeline)
  - [Step 1: Generate Pairs](#step-1-generate-pairs-pos_negpy)
  - [Step 2: Convert to NPY](#step-2-convert-to-npy-convert_to_npypy)
  - [Step 3: Evaluate](#step-3-evaluate-test1py)
- [Supported Models](#supported-models)
- [Per-Twin-Pair Analysis](#per-twin-pair-analysis)
- [Acknowledgement](#acknowledgement)

## Environment

### Dependencies

- PyTorch
- NumPy
- OpenCV (`cv2`)
- scikit-learn
- matplotlib
- tqdm

## Dataset Preparation

### Input Structure

Organize your dataset into 10 folds with twin pair subfolders, each containing two individual twin folders with 12 aligned images:

```
Celeb_twins_folds/
├── fold_00/
│   ├── Olsen_twins/
│   │   ├── Ashley_Olsen/
│   │   │   ├── 001.jpg
│   │   │   ├── 002.jpg
│   │   │   └── ... (12 images)
│   │   └── Mary-Kate_Olsen/
│   │       └── ... (12 images)
│   ├── Sprouse_twins/
│   │   ├── Cole_Sprouse/
│   │   └── Dylan_Sprouse/
│   └── ... (5 twin pairs per fold)
├── fold_01/
└── ... (10 folds)
```

### Training Sets

CTTS is evaluated using pre-trained models. We support weights trained on:

- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
- [WebFace4M](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

## Pipeline

## Supported Models

CTTS can be evaluated with any model compatible with the [SOTA-FR-train-and-test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) framework:

| Method | Conference | Test Flag |
|---|---|---|
| ArcFace | CVPR19 | `--add_flip` |
| MagFace | CVPR21 | `--add_flip` |
| AdaFace | CVPR22 | `--add_norm` |
| UniFace | ICCV23 | `--add_flip` |


## Acknowledgement

This work builds on the [SOTA-FR-train-and-test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) framework by Haiyu Wu et al. We thank the authors for their valuable contributions to the face recognition community.

## License

MIT
