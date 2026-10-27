"""Train the MNIST CNN and VAE. Run: python train_models.py [--classifier-epochs 3 --vae-epochs 10]"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MNISTClassifier, MNISTVAE, vae_loss

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--vae-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", default=PROJECT_DIR / "models")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Training on {device}; MNIST has {len(train_data):,} training images.")

    classifier = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    classifier.train()
    for epoch in range(args.classifier_epochs):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"Classifier epoch {epoch + 1}/{args.classifier_epochs}: loss {total_loss / len(train_data):.4f}")
    torch.save(classifier.cpu().state_dict(), output / "classifier.pt")

    vae = MNISTVAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for epoch in range(args.vae_epochs):
        total_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = vae(images)
            loss = vae_loss(reconstruction, images, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"VAE epoch {epoch + 1}/{args.vae_epochs}: loss {total_loss / len(train_data):.2f}")
    torch.save(vae.cpu().state_dict(), output / "vae.pt")
    print(f"Saved trained models to {output.resolve()}")


if __name__ == "__main__":
    main()
