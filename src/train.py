# src/train.py
import torch
import torch.nn as nn
from src.utils import add_noise


def train(model, train_loader, test_loader, cfg):


    tcfg = cfg.train

    device = torch.device(tcfg.device)
    epochs = int(tcfg.epochs)
    lr = float(tcfg.lr)
    noise_std = float(getattr(tcfg, "noise_std", 0.2))
    save_path = str(getattr(tcfg, "save_path", "dae_best.pth"))

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler configurable
    step_size = int(getattr(tcfg, "scheduler_step_size", 10))
    gamma = float(getattr(tcfg, "scheduler_gamma", 0.5))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for x_clean, _ in train_loader:
            x_clean = x_clean.to(device)

            # bruit variable pendant le train
            cur_std = noise_std * torch.rand(1).item()
            x_noisy = add_noise(x_clean, cur_std)

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

                # bruit FIXE en test
                x_noisy = add_noise(x_clean, noise_std)

                x_recon = model(x_noisy)
                loss = criterion(x_recon, x_clean)

                test_loss += loss.item()

        test_loss /= len(test_loader)

        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train MSE: {train_loss:.4f} | "
            f"test MSE: {test_loss:.4f} | "
            f"best: {best_loss:.4f}"
        )

    print(f"Best model saved as {save_path}")

    return {
        "best_test_loss": best_loss,
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
    }
