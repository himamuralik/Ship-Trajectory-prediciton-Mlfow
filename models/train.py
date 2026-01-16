python
import torch
import torch.nn as nn

def train_model(model, dataloader, optimizer, epochs, device):
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

