# U-SamNet

Official PyTorch implementation of **U-SamNet: A Multi-Scale Segmentation Approach for Breast Tumor Ultrasound Images**.

U-SamNet is a fully automatic segmentation framework that coordinates global structural priors from a frozen Segment Anything Model (SAM) image encoder with local texture and boundary features learned by a CNN. The implementation includes lightweight residual SAM adaptation, multi-scale prior projection, reliability-aware prior fusion, contrast-aware dynamic boundary Hadamard attention, auxiliary boundary supervision, and residual correction.

## Highlights

- Fully automatic inference without point, box, or generated prompts.
- Frozen SAM image encoder with lightweight residual adaptation.
- Multi-scale interaction between SAM-derived priors and CNN features.
- Reliability-aware fusion in ambiguous and low-contrast regions.
- Registered configurations for the ablation studies reported in the manuscript.

## Repository structure

```text
U-SamNet/
├── configs/                 # Dataset, training, and ablation configurations
├── datasets/                # Image-mask dataset loader
├── docs/                    # Data preparation notes
├── models/                  # U-SamNet and SAM image encoder
│   └── sam_pretrain/        # Place the pretrained SAM checkpoint here
├── scripts/                 # Example training and inference commands
├── engine.py                # Training and evaluation loops
├── train.py                 # Training entry point
├── test.py                  # Inference and metric evaluation
└── utils.py                 # Losses, transforms, metrics, and visualization
```

Datasets, model checkpoints, predictions, and experiment outputs are intentionally excluded from the repository.

## Installation

Python 3.9 or later is recommended.

```bash
git clone https://github.com/tubixiansheng/U-SamNet.git
cd U-SamNet

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA environment if the default package from `requirements.txt` is not suitable. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## SAM checkpoint

The released configuration uses the SAM ViT-B image encoder. Download the official checkpoint and place it at:

```text
models/sam_pretrain/sam_vit_b_01ec64.pth
```

```bash
wget -P models/sam_pretrain \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The SAM checkpoint is not redistributed in this repository. The model raises a clear error if the requested checkpoint is missing.

## Data preparation

Prepare each dataset with the following structure:

```text
DATASET_ROOT/
├── train/
│   ├── images/
│   └── masks/       # use labels/ for BUSI
└── val/
    ├── images/
    └── masks/       # use labels/ for BUSI
```

Images and masks must share the same filename stem. A mask may also use the suffix `_mask`, for example `case_001.png` and `case_001_mask.png`. Supported image formats include PNG, JPEG, BMP, and TIFF.

The registered dataset names are `BUSI`, `MosMedData`, `SZU-BCH-BUS2093`, `SZU-BCH-BUS2744`, and `SZU-BCH-TUS983`. Dataset files are not included. Additional details are available in [docs/DATASETS.md](docs/DATASETS.md).

## Training

The default configuration follows the manuscript setting with 256 x 256 inputs, AdamW, a learning rate of `1e-4`, a weight decay of `1e-4`, batch size 1, and 140 epochs.

```bash
python train.py \
  --dataset BUSI \
  --dataset-root /path/to/BUSI \
  --gpu-id 0 \
  --epochs 140 \
  --batch-size 1
```

Training artifacts are written below `results/` unless `--work-dir` is supplied. Resume training by leaving `latest.pth` in the selected run's `checkpoints/` directory.

For a quick pipeline check with a prepared dataset:

```bash
python train.py \
  --dataset BUSI \
  --dataset-root /path/to/BUSI \
  --gpu-id 0 \
  --smoke-test
```

## Inference and evaluation

Run inference on a single image or an image directory:

```bash
python test.py \
  --dataset BUSI \
  --weights /path/to/best.pth \
  --input /path/to/BUSI/val/images \
  --output-dir outputs/BUSI \
  --gpu-id 0
```

Provide ground-truth masks to compute IoU, DSC, accuracy, precision, specificity, and sensitivity. Threshold scanning is optional.

```bash
python test.py \
  --dataset BUSI \
  --weights /path/to/best.pth \
  --input /path/to/BUSI/val/images \
  --mask-dir /path/to/BUSI/val/labels \
  --output-dir outputs/BUSI \
  --scan-thresholds \
  --summary-json outputs/BUSI/summary.json
```

## Ablation configurations

The default `current_best` configuration corresponds to the released U-SamNet setup. Paper ablations are registered in `configs/ablation_registry.py` and can be selected with `--ablation`.

```bash
python train.py \
  --dataset BUSI \
  --dataset-root /path/to/BUSI \
  --ablation no_sam_adapter
```

Other examples include `no_confidence_fusion`, `no_boundary_branch`, `no_uncertainty_supervision`, `no_deep_supervision`, `no_mask_branch_supervision`, `static_cdbha`, `no_cdbha_cab`, `no_cdbha_boundary_gate`, `no_cdbha_dynamic_routing`, `se_attention`, `cbam_attention`, `axial_attention`, and `residual_shared_sam_aux`.

## Reproducibility notes

- Random seeds are fixed in the training entry point.
- The SAM encoder is frozen by default.
- Evaluation uses a binary threshold of 0.5 unless overridden.
- `test.py` computes all binary metrics from the same accumulated confusion counts, so the reported IoU and DSC are mutually consistent.
- Dataset splits and preprocessing must match the experimental protocol when reproducing manuscript results.

## Citation

If this code is useful in your research, please cite the manuscript:

```bibtex
@misc{lai2026usamnet,
  title  = {U-SamNet: A Multi-Scale Segmentation Approach for Breast Tumor Ultrasound Images},
  author = {Zhihui Lai and Qiwei Wu and Heng Kong and Linlin Shen},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/tubixiansheng/U-SamNet}
}
```

## Acknowledgments

This implementation uses the image-encoder design and pretrained weights released with [Segment Anything](https://github.com/facebookresearch/segment-anything). The local segmentation pathway was developed from the [EGE-UNet](https://github.com/JCruan519/EGE-UNet) codebase. We thank the authors of these projects for making their work publicly available. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## License

This project is released under the [Apache License 2.0](LICENSE). Users are responsible for complying with the licenses and terms of any external datasets and pretrained weights.
