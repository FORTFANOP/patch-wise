import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# importing from local classes
from dataset import carsData, randomPatchify
from model import ConvAutoencoder

def main():
    path = '../data/cars'
    path_list = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path_list.append(os.path.join(root, file))

    print(f"Found {len(path_list)} images.")


    base_transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor()
    ])

    # 3. Initialize dataset and splits
    dataset = carsData(
        img_paths=path_list,
        base_transform=base_transform,
        input_transform=randomPatchify(num_patches=(10, 20), patch_size=(10, 40), p=1.0)
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # create instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for x_corrupted, x_clean in train_loader:
            x_corrupted, x_clean = x_corrupted.to(device), x_clean.to(device)

            optimizer.zero_grad()
            pred = model(x_corrupted)
            loss = criterion(pred, x_clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.6f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_corrupted, x_clean in val_loader:
            x_corrupted, x_clean = x_corrupted.to(device), x_clean.to(device)
            pred = model(x_corrupted)
            loss = criterion(pred, x_clean)
            val_loss += loss.item()

    print(f"Final Validation Loss: {val_loss / len(val_loader):.6f}")

    # TODO: Save the model weights
    # torch.save(model.state_dict(), "patchwise_autoencoder.pth")

if __name__ == "__main__":
    main()