# BraTS2024

3D segmentation of post-treatment adult gliomas on BraTS 2024 multi-parametric MRI
(T1, T1-Gd, T2, FLAIR), built with PyTorch and MONAI.

## Requirements

- Linux (tested on <system/cluster>)
- NVIDIA GPU with CUDA support (training was done on an NVIDIA A40, 48 GB)
- [Miniforge](https://github.com/conda-forge/miniforge) or another conda distribution
- ~<X> GB of free disk space for the raw data and preprocessing cache
- A [Synapse](https://www.synapse.org/) account with access to the BraTS 2024 data

## Installation

```bash
git clone https://github.com/AleczO/BraTS2024.git
cd BraTS2024

conda env create -f environment.yml
conda activate brats
```

PyTorch is installed separately, because the right build depends on your GPU driver.
Check the CUDA version supported by your driver with `nvidia-smi`, then pick the matching
command from [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

1. Register for the BraTS 2024 Adult Glioma Post-Treatment challenge on Synapse
   and accept the data use terms.
2. Create a Synapse personal access token and put it in a `.env` file in the repo root:

```
   SYNAPSE_TOKEN=<your-token>
```

3. Download the data:

```bash
   python scripts/download.py
```

By default, data is stored in `data/` and training outputs in `outputs/`.
To use other locations (e.g. a scratch disk on a cluster), set:

```bash
export DATA_DIR=/path/to/data
export OUTPUT_DIR=/path/to/outputs
```

The BraTS data cannot be redistributed and is not included in this repository.
BraTS 2024 was used because the BraTS 2025 dataset had not yet been publicly
released at the time of this work.


## Citation

If you use this code with the BraTS 2024 data, please cite:

- M. Correia de Verdier et al., "The 2024 Brain Tumor Segmentation (BraTS)
  Challenge: Glioma Segmentation on Post-treatment MRI", arXiv:2405.18368, 2024.

<!-- TODO: verify the full list of required citations on the BraTS 2024 Synapse page -->
