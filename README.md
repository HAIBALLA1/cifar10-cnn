# Denoising Autoencoder on CIFAR-10

## Project Overview

This project implements a Convolutional Denoising Autoencoder trained on the CIFAR-10 dataset.

The objective is to reconstruct clean images from noisy inputs using deep learning techniques. 
The model learns to remove Gaussian noise while preserving the structure and color distribution of images.

---

## Dataset

Dataset used: **CIFAR-10**

- 60,000 RGB images
- Image size: 32 × 32
- 10 object classes
- 50,000 training images
- 10,000 test images

### Normalization

Images are normalized using CIFAR-10 statistics:

Mean = (0.4914, 0.4822, 0.4465) 
Std  = (0.2023, 0.1994, 0.2010)

### Data Augmentation (Training Only)

- Random horizontal flip 
- Random crop (with padding)

---

## Model Architecture

The model follows an encoder–decoder structure.

### Encoder
- Conv2D (3 → 32) + BatchNorm + ReLU 
- Conv2D (32 → 32) + BatchNorm + ReLU 
- MaxPool (32×32 → 16×16) 
- Conv2D (32 → 64) + BatchNorm + ReLU 
- MaxPool (16×16 → 8×8)

### Decoder
- ConvTranspose2D (64 → 32) (8×8 → 16×16) 
- ConvTranspose2D (32 → 32) (16×16 → 32×32) 
- Conv2D (32 → 3)

The encoder compresses spatial information, and the decoder reconstructs the original image resolution using transpose convolutions.

---

## Training Procedure

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning Rate Scheduler: StepLR
- Best model checkpointing based on validation loss

### Noise Strategy

- Dynamic Gaussian noise during training 
- Fixed noise level during evaluation 

This improves generalization and model robustness.

---

## Evaluation

During evaluation, the model displays:

- Noisy images 
- Reconstructed images 
- Clean images 

Images are properly denormalized for visualization.

---

## Results

After training for 30 epochs:

- Final Train MSE: 0.0446 
- Final Test MSE: 0.0459 
- Best Test MSE: 0.0459 (saved as `dae_best.pth`) 

### Robustness to Noise (Benchmark)

The trained model was evaluated under different Gaussian noise levels:

| Noise Std | Test MSE | PSNR (dB) |
|------------|----------|------------|
| 0.05 | 0.0403 | 13.95 |
| 0.10 | 0.0414 | 13.83 |
| 0.20 | 0.0458 | 13.39 |
| 0.30 | 0.0533 | 12.73 |

The model remains stable under moderate noise levels and degrades gracefully as noise intensity increases.

### Example Reconstruction

![Reconstruction Example](images/recon_grid.png)

---

## Technical Highlights

- Encoder–Decoder CNN architecture 
- Batch Normalization for stable training 
- Learning rate scheduling 
- Best model checkpointing 
- Dynamic noise augmentation 
- Proper image denormalization for visualization 

---

## Project Structure

```
src/
 ├── data.py        # Data loading and preprocessing
 ├── model.py       # Autoencoder architecture
 ├── train.py       # Training loop and checkpointing
 ├── eval.py        # Evaluation and visualization
 ├── utils.py       # Utility functions
scripts/
 ├── benchmark.py   # Performance benchmarking
 ├──run_train.py
```

---

## Why This Project?

Image denoising is a fundamental computer vision task used in:

- Photography enhancement 
- Medical imaging 
- Image preprocessing pipelines 

This project demonstrates practical implementation of convolutional autoencoders in PyTorch, including training optimization and robustness evaluation.

---

## Future Improvements

- Add SSIM metric
- Implement U-Net architecture with skip connections 
- Add perceptual loss 
- Experiment with different noise types 
- Train on higher resolution datasets 
- Integrate TensorBoard logging 

---

## Skills Demonstrated

- Deep Learning with PyTorch 
- CNN-based Autoencoders 
- Image preprocessing and normalization 
- Training pipeline design 
- Learning rate scheduling 
- Model checkpointing 
- Performance benchmarking 
- Experiment reproducibility 

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python run_train.py
```

### Benchmark the model

```bash
python -m scripts.benchmark
```

### Evaluate visual reconstructions

```bash
python -m src.eval
```

The best model checkpoint will be saved as:

```
dae_best.pth
```

