# Learning Informative Invariant Representations via Hierarchical Latent Decomposition

This repository is the official implementation of "**[Learning Informative Invariant Representations via Hierarchical Latent Decomposition](https://neurips.cc/virtual/2026/loc/sydney/poster/155712)**".

**BLENDER** learns domain-invariant and domain-specific representations through hierarchical latent decomposition, combining classification, reconstruction, and independence regularization for domain generalization.

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

Supported datasets: **RotatedMNIST, VLCS, PACS, OfficeHome, DomainNet, and TerraIncognita**.

Download and prepare all datasets under `./datasets/`:

```bash
python datasets/download.py
```

## Training

Run from the repository root:

```bash
python analysis/model_train.py --model_name blender --dataset_name PACS --seed 42
```

This trains across all held-out domains. Configurations are in `methods/configs/<model_name>.json`, and checkpoints are saved to `analysis/models/<model_name>/`.

## Evaluation

Set `data_list`, `model_list`, and `seeds` in `analysis/performance.py` to match your saved checkpoints, then run:

```bash
python analysis/performance.py
```

Results are saved to `analysis/results/`.

## Analysis

- `analysis/plot_latent.py`: latent-space visualization.
- `analysis/plot_reconstruction.py`: RotatedMNIST reconstruction.

Adjust experiment selections in the scripts before use. Figures are saved to `analysis/figures/`.

## Citation

TODO: Add BibTeX.

## License

[MIT](LICENSE)
