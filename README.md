# Generative vs. Discriminative Model Lab

An interactive Streamlit application for drawing or uploading a handwritten digit, classifying it with a CNN, and generating MNIST-like samples with a trained variational autoencoder (VAE). It uses the trained model weights in `models/` and does not present random noise as generated handwriting.

## Requirements

- Python 3.12, 64-bit (select Python 3.12 in Streamlit Community Cloud Advanced settings)
- Packages pinned in `requirements.txt`
- `models/classifier.pt` and `models/vae.pt` in the repository

## Local setup (Windows PowerShell)

From this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8505
```

Open `http://localhost:8505`. The app resolves model paths relative to `app.py`, so this also works from a Linux checkout.

## Train both models

The existing trained weights are included for running and deployment; retraining is unnecessary. To intentionally train or replace them locally, run:

```powershell
.\.venv\Scripts\python.exe train_models.py
```

This downloads MNIST into `data/`, trains the CNN (3 epochs by default) and VAE (10 epochs), and saves `models/classifier.pt` and `models/vae.pt`. Training overwrites those two files, so do not run it if you want to preserve the supplied weights.

## Use the app

1. Draw a single digit on the white canvas with a black stroke, or choose **Upload** and select a PNG, JPEG, BMP, or WebP image.
2. Select **Predict digit** to view the normalized 28 x 28 input, predicted class, softmax confidence, and probabilities for all ten classes.
3. Select **Generate new digits** to sample images from the trained VAE. Adjust the sample count with the slider.

Inputs are oriented, converted to grayscale, polarity-normalized, cropped around visible ink, resized, centered, and scaled to `[0, 1]` for the CNN.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository. Include `models/classifier.pt` and `models/vae.pt`; the `.gitignore` intentionally excludes training data and local environments but does not ignore model weights.
2. In Streamlit Community Cloud, create an app and select your repository and branch. Set the main file path to `app.py`.
3. Open **Advanced settings**, select Python **3.12**, then deploy. Community Cloud installs dependencies from the root `requirements.txt`.
4. Open the URL provided by Community Cloud after its build completes.

The model files are about 1.7 MB and 2.6 MB, respectively, and are kept as normal Git files. Do not commit `.streamlit/secrets.toml`, `.env` files, virtual environments, or downloaded MNIST data.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, preprocessing, model loading, prediction, and generation |
| `models.py` | CNN and VAE architectures shared by app and training script |
| `train_models.py` | MNIST download and training script |
| `requirements.txt` | Pinned Python dependencies |
| `models/` | Trained classifier and VAE state dictionaries |
| `data/` | Downloaded MNIST training data (ignored by Git) |

## Model overview

The CNN is discriminative: it maps an input image to one of ten digit labels. The VAE is generative: its encoder learns a latent distribution, and its decoder maps sampled latent vectors back to digit images.
