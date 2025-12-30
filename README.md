# RetinexFormer with MS-SSIM and Perceptual Loss

This is a modified version of [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer) that implements additional loss functions for low-light image enhancement, specifically tested on the **LOL v1 dataset**.

## Modifications

This repository extends the original RetinexFormer with:

- **MS-SSIM Loss**: Multi-Scale Structural Similarity loss based on "Loss Functions for Image Restoration with Neural Networks" (Zhao et al.)
- **Perceptual Loss**: VGG19-based perceptual loss inspired by EnlightenGAN
- **Three Training Configurations**:
  1. **Baseline**: L1 loss only (original RetinexFormer)
  2. **L1 + Perceptual**: L1 (weight=1.0) + Perceptual Loss (weight=0.1)
  3. **L1 + MS-SSIM Mix**: Mix Loss with α=0.84 for L1 and (1-α)=0.16 for MS-SSIM

## Requirements

- Python 3.9+
- PyTorch 2.6.0+ with CUDA 12.4+
- NVIDIA GPU with CUDA support (tested on RTX 5060 Ti)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tamnd04/Low-Light-Image-Enhancement-Project.git
cd Low-Light-Image-Enhancement-Project
```

2. Create conda environment:
```bash
conda create -n torch2 python=3.9 -y
conda activate torch2
```

3. Install dependencies:
```bash
# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install numpy opencv-python tqdm pyyaml tensorboard scikit-image lpips
```

4. Download LOL v1 dataset and organize as follows:
```
data/
└── LOLv1/
    ├── Train/
    │   ├── input/
    │   └── target/
    └── Test/
        ├── input/
        └── target/
```

You can download LOL v1 from:
- [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)
- [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`)

## Training

### Option 1: Baseline (L1 Loss Only)
```bash
python basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml
```

### Option 2: L1 + Perceptual Loss
```bash
python basicsr/train.py --opt Options/RetinexFormer_LOL_v1_Perceptual.yml
```

### Option 3: L1 + MS-SSIM Mix Loss
```bash
python basicsr/train.py --opt Options/RetinexFormer_LOL_v1_MSSSIM.yml
```

Training outputs:
- **Models**: `experiments/{experiment_name}/models/`
- **Metrics**: `experiments/{experiment_name}/metric.csv`
- **TensorBoard logs**: `tb_logger/{experiment_name}/`

Each configuration trains for 150K iterations (~33 hours on RTX 5060 Ti).

## Testing

Test on LOL v1 test set:

```bash
cd Enhancement

# Test baseline model
python test_from_dataset.py --opt ../Options/RetinexFormer_LOL_v1.yml --weights ../experiments/RetinexFormer_LOL_v1/models/net_g_latest.pth --dataset LOLv1

# Test perceptual model
python test_from_dataset.py --opt ../Options/RetinexFormer_LOL_v1_Perceptual.yml --weights ../experiments/RetinexFormer_LOL_v1_Perceptual/models/net_g_latest.pth --dataset LOLv1

# Test MS-SSIM model
python test_from_dataset.py --opt ../Options/RetinexFormer_LOL_v1_MSSSIM.yml --weights ../experiments/RetinexFormer_LOL_v1_MSSSIM/models/net_g_latest.pth --dataset LOLv1
```
- #### Self-ensemble testing strategy
We added the self-ensemble strategy in the testing code to derive better results. Just add a `--self_ensemble` action at the end of the above test command to use it.

After having trained all three of the models, you can also plot the training loss curve:
```bash
python plot_training_curves.py
```

## Results Comparison

| Configuration | Loss Function | PSNR (dB) | SSIM | Notes |
|--------------|---------------|-----------|-------|-------|
| Baseline | L1 only | 25.154 | 0.845 |Original RetinexFormer |
| Perceptual | L1 + Perceptual (0.1) | 24.406 | 0.860 | VGG19 layers: conv3_4, conv4_4, conv5_4 |
| MS-SSIM | L1 + MS-SSIM Mix (0.84/0.16) | 24.058 | 0.840 | Following Zhao et al. paper |

*(Results based on LOL v1 test set with 15 test images)*

## Loss Function Details

### 1. Baseline (L1 Loss)
```yaml
pixel_opt:
  type: L1Loss
  loss_weight: 1.0
  reduction: mean
```

Standard L1 loss applied to all intermediate predictions from the model.

### 2. Perceptual Loss
```yaml
pixel_opt:
  type: L1Loss
  loss_weight: 1.0

perceptual_opt:
  type: PerceptualLoss
  layer_weights:
    'conv3_4': 1.0
    'conv4_4': 1.0
    'conv5_4': 1.0
  vgg_type: vgg19
  perceptual_weight: 0.1
  criterion: l1
```

**Total Loss**: `L = L1 + 0.1 × Perceptual`

Based on EnlightenGAN's approach using VGG19 features from layers conv3_4, conv4_4, and conv5_4.

### 3. MS-SSIM Mix Loss
```yaml
pixel_opt:
  type: L1Loss
  loss_weight: 0.84  # α from paper

msssim_opt:
  type: MS_SSIMLoss
  loss_weight: 0.16  # (1-α) from paper
  data_range: 1.0
  size_average: true
```

**Total Loss**: `L = 0.84 × L1 + 0.16 × (1 - MS-SSIM)`

Follows the Mix Loss formulation from "Loss Functions for Image Restoration with Neural Networks" (Zhao et al.), where:
- α = 0.84 (weight for L1 loss)
- (1-α) = 0.16 (weight for MS-SSIM loss)
- Weights sum to 1.0 for proper balance

The MS-SSIM computes multi-scale structural similarity using 5 pyramid levels with weights [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].

## Project Structure

```
.
├── basicsr/              # Core training and model code
│   ├── train.py         # Training script (with tqdm progress bar)
│   ├── models/
│   │   ├── image_restoration_model.py  # Model with multi-loss support
│   │   └── losses/
│   │       ├── perceptual_loss.py      # VGG19 perceptual loss
│   │       └── msssim_loss.py          # MS-SSIM loss implementation
│   └── data/            # Dataset loaders
├── Enhancement/         # Testing scripts
│   └── test_from_dataset.py
├── Options/             # Training configurations
│   ├── RetinexFormer_LOL_v1.yml           # Baseline
│   ├── RetinexFormer_LOL_v1_Perceptual.yml # L1 + Perceptual
│   └── RetinexFormer_LOL_v1_MSSSIM.yml     # L1 + MS-SSIM Mix
├── data/                # Dataset folder (not included in repo)
└── experiments/         # Training outputs (not included in repo)
```

## Key Implementation Details

### Training Optimizations
- **Batch size**: 16
- **Workers**: 8 with CUDA prefetching
- **Mixed precision**: Disabled for stability
- **Progress tracking**: tqdm progress bar added to training loop
- **GPU utilization**: ~95% during training

### Loss Function Implementation
For MS-SSIM Mix Loss, the implementation correctly follows the paper by:
1. Applying both L1 and MS-SSIM to the **final output only** (not intermediate predictions)
2. Using normalized weights that sum to 1.0 (α + (1-α) = 1.0)
3. Computing MS-SSIM on the same prediction as L1 for proper balancing

This differs from the baseline, where L1 is applied to all intermediate predictions.

## References

- **Original RetinexFormer**: Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement", ICCV 2023
  - GitHub: https://github.com/caiyuanhao1998/Retinexformer
- **MS-SSIM Loss**: Zhao et al., "Loss Functions for Image Restoration with Neural Networks", IEEE TCI 2017
- **Perceptual Loss**: Jiang et al., "EnlightenGAN: Deep Light Enhancement without Paired Supervision", TIP 2021

## Citation

If you use this code, please cite the original RetinexFormer paper:

```bibtex
@InProceedings{Cai_2023_ICCV,
    author    = {Cai, Yuanhao and Bian, Hao and Lin, Jing and Wang, Haoqian and Timofte, Radu and Zhang, Yulun},
    title     = {Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12504-12513}
}
```

## License

This project is released under the same license as the original RetinexFormer (MIT License).

## Acknowledgments

This work is based on [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer) by Yuanhao Cai et al. We extend our gratitude to the original authors for their excellent work and open-source contribution.

