# src/train.py
import torch
import torch.nn as nn
from src.utils import add_noise



def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs=10,
    lr=1e-3,
    noise_std=0.2,
):
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for x_clean, _ in train_loader:
            x_clean = x_clean.to(device)

            x_noisy = add_noise(x_clean, noise_std)

            optimizer.zero_grad()
            x_recon = model(x_noisy)
            loss = criterion(x_recon, x_clean)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== TEST =====
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for x_clean, _ in test_loader:
                x_clean = x_clean.to(device)
                x_noisy = add_noise(x_clean, noise_std)

                x_recon = model(x_noisy)
                loss = criterion(x_recon, x_clean)

                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train MSE: {train_loss:.4f} | "
            f"test MSE: {test_loss:.4f}"
        )

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "dae.pth")
    print("Model saved as dae.pth")
