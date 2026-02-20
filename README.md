# Robust Convolutional Denoising Autoencoder on CIFAR-10
Hydra • MLflow • Docker • PyTorch

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Hydra](https://img.shields.io/badge/Hydra-Config--Driven-6f42c1)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-0194E2)
![Docker](https://img.shields.io/badge/Docker-Reproducible-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project implements a Convolutional Denoising Autoencoder (DAE) trained on CIFAR-10 to reconstruct clean images from Gaussian-corrupted inputs.

It combines:

- Deep Learning (PyTorch)
- Config-driven experiments (Hydra)
- Full experiment tracking (MLflow)
- Robustness benchmarking
- Reproducible execution (Docker)

The focus is not only on model performance, but also on clean ML engineering practices.

---

## Architecture Diagram

```
        +-------------------+
        |   Noisy Image     |
        |   (32x32x3)       |
        +-------------------+
                 │
                 ▼
         ┌────────────────┐
         │    Encoder     │
         │ Conv + BN + ReLU
         │ MaxPool layers │
         └────────────────┘
                 │
                 ▼
           Latent Space
                 │
                 ▼
         ┌────────────────┐
         │    Decoder     │
         │ ConvTranspose  │
         │ + Conv         │
         └────────────────┘
                 │
                 ▼
        +-------------------+
        | Reconstructed Img |
        +-------------------+
```

---

## Dataset

Dataset: CIFAR-10

- 60,000 RGB images
- 32 × 32 resolution
- 10 object classes
- 50,000 training images
- 10,000 test images

Normalization:

Mean = (0.4914, 0.4822, 0.4465) 
Std  = (0.2023, 0.1994, 0.2010)

Training augmentation:

- Random horizontal flip
- Random crop with padding

---

## Training

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Scheduler: StepLR
- Best model checkpointing
- Seed control for reproducibility
- Dynamic Gaussian noise injection during training
- Fixed noise during evaluation

Train locally:

```bash
python run_train.py
```

---

## Evaluation

Evaluate and log results to MLflow:

```bash
python -m src.eval train.save_path=dae_best.pth
```

Launch MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

http://127.0.0.1:5000

---

## Robustness Benchmark

Evaluate model across multiple Gaussian noise levels:

```bash
python -m scripts.benchmark train.save_path=dae_best.pth
```

Outputs:

- CSV metrics table
- Reconstruction grid
- MLflow logged metrics

---

## Docker Usage

Train:

```bash
docker compose up --build dae
```

Launch MLflow:

```bash
docker compose up --build mlflow
```

Open:

http://localhost:5000

Evaluate with Docker:

```bash
docker compose run --rm dae python -m src.eval train.save_path=dae_best.pth
```

Benchmark with Docker:

```bash
docker compose run --rm dae python -m scripts.benchmark train.save_path=dae_best.pth
```

---

## Project Structure

```
src/
 ├── data.py
 ├── model.py
 ├── train.py
 ├── eval.py
 ├── utils.py

scripts/
 ├── benchmark.py

configs/
 ├── config.yaml
 ├── model/
 ├── data/
 ├── train/

run_train.py
docker-compose.yml
Dockerfile
```

---

## Hydra Overrides

Change noise level:

```bash
python run_train.py train.noise_std=0.2
```

Change epochs:

```bash
python run_train.py train.epochs=50
```

---

## Why This Project Matters

Image denoising is widely used in:

- Medical imaging
- Photography enhancement
- Preprocessing for computer vision systems

This repository demonstrates:

- Deep learning modeling
- Robustness evaluation
- Reproducibility
- Clean experiment management

---

## Skills Demonstrated

- PyTorch Deep Learning
- CNN Autoencoders
- Data preprocessing and normalization
- Learning rate scheduling
- Model checkpointing
- Robustness benchmarking
- Hydra configuration management
- MLflow experiment tracking
- Docker containerization
- Reproducible ML pipelines

---

## Future Improvements

- Add SSIM metric
- Implement U-Net with skip connections
- Add perceptual loss
- Export to ONNX / TorchScript
- Add CI pipeline
