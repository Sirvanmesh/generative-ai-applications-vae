# Generative AI Applications: Convolutional VAE on Fashion-MNIST

Udacity Master's in AI Capstone, Project 5.

## What it does
- Trains a **convolutional Variational Autoencoder (VAE)** in PyTorch that generates new 28x28 grayscale clothing images and learns a smooth latent space of clothing styles.
- Evaluates the outputs with reconstructions, 64 random samples, latent interpolations, per-class error and a memorisation check (nearest training neighbours).
- Includes a **Generative AI Analysis and Ethics Report** (`module_summary.pdf`) covering model design, output evaluation, bias, creative ownership and misuse.

## Dataset
**Fashion-MNIST** (Xiao et al., 2017): 70,000 real Zalando product photos in 10 classes, MIT License.
Source: https://github.com/zalandoresearch/fashion-mnist

You do not need to download anything by hand. The notebook calls `torchvision.datasets.FashionMNIST(root="./data", download=True)`, which downloads the data (about 30 MB) into `./data` on the first run.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## How to run
```bash
jupyter notebook generative_model.ipynb
```
Run all cells. The notebook uses a GPU if one is available, otherwise the CPU (about 5 minutes in total on a CPU). Seeds are fixed, so the results are reproducible.

To rebuild the PDF report from the notebook results: `python make_report.py`.

## Main outputs
| Path | Contents |
|---|---|
| `generative_model.ipynb` | Full pipeline with saved outputs, ending with a Summary |
| `module_summary.pdf` | Generative AI Analysis and Ethics Report |
| `figures/` | Data samples, loss curves, reconstructions, generated samples, interpolation, per-class error, nearest neighbours, beta-VAE comparison, t-SNE |
| `results/` | `metrics.json` and per-epoch training histories |
| `models/` | Trained weights (`vae_fashionmnist.pt`, 6.8 MB) |
