import os
import time
import torch
import torch.nn as nn
from src.return_ssim import return_ssim

def train_autoencoder(model, train_loader, test_loader, optimizer, scheduler, num_epochs,
                      criterion=nn.MSELoss(), DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
                      ,clip_value=5.0, patience=4, model_name="model", save_dir="../models"):

    start = time.time()
    model = model.to(DEVICE)
    best_ssim = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        count = 0
            
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            running_loss += loss.item()
            count += 1

        ssim_score = return_ssim(model, test_loader)
        avg_loss = running_loss / count
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, SSIM: {ssim_score:.4f}")

        if ssim_score > best_ssim:
            best_ssim = ssim_score
            patience_counter = 0
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"{model_name}.pt")
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best SSIM: {best_ssim:.4f}")
                break

    end = time.time()
    print(f"Training time: {end - start:.2f}s")