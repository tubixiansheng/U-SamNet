# Dataset preparation

U-SamNet expects an explicit training and validation split. The repository does not redistribute medical images or annotations.

## Directory layout

```text
DATASET_ROOT/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

For BUSI, use `labels/` instead of `masks/`. You may override the registered root with `--dataset-root` during training. In inference, pass the image location with `--input` and, when available, the annotation location with `--mask-dir`.

The loader pairs files by stem and also recognizes an optional `_mask` suffix. For example, both of the following pairs are valid:

```text
images/case_001.png  masks/case_001.png
images/case_002.png  masks/case_002_mask.png
```

## Registered datasets

| CLI name | Modality | Default mask directory |
| --- | --- | --- |
| `BUSI` | Breast ultrasound | `labels` |
| `MosMedData` | Lung CT | `masks` |
| `SZU-BCH-BUS2093` | Breast ultrasound | `masks` |
| `SZU-BCH-BUS2744` | Breast ultrasound | `masks` |
| `SZU-BCH-TUS983` | Thyroid ultrasound | `masks` |

The two SZU-BCH breast ultrasound datasets will be released for research use following acceptance of the manuscript. Users should obtain public datasets from their original providers and comply with their licenses and terms of use.

## Adding another dataset

Add a dataset entry and normalization statistics to `configs/config_setting.py`, then add the matching canonical name to `myNormalize` in `utils.py`. Image and mask interpolation are fixed to bilinear and nearest-neighbor interpolation, respectively.
