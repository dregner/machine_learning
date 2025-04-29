import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Process

# === Dataset ===
class CarRacingDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, stack_size=4):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.stack_size = stack_size

        self.frames_by_lap = {}
        for item in self.data:
            lap = item['lap']
            self.frames_by_lap.setdefault(lap, []).append(item)

        for lap in self.frames_by_lap:
            self.frames_by_lap[lap].sort(key=lambda x: x['step'])

        self.samples = []
        for frames in self.frames_by_lap.values():
            for i in range(stack_size - 1, len(frames)):
                self.samples.append(frames[i - stack_size + 1 : i + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_seq = self.samples[idx]
        stacked = []
        for frame in frame_seq:
            f = frame["observation_path"].split("\\")
            img = Image.open(f[0]+'/'+f[1]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            stacked.append(img)
        image_stack = torch.cat(stacked, dim=0)

        steer = frame_seq[-1]['action'][0]
        gas = frame_seq[-1]['action'][1]
        brake = frame_seq[-1]['action'][2]

        gas_brake = gas - brake

        action = torch.tensor([steer, gas_brake], dtype=torch.float32)
        return image_stack, action

# === Model ===
class SimpleCarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        return self.fc(x)

# === Training function ===
def train_model(batch_size, learning_rate, epochs, data_file, output_dir):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    dataset = CarRacingDataset(data_file, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCarCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, actions in train_loader:
            images, actions = images.to(device), actions.to(device)
            preds = model(images)
            loss = loss_fn(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, actions in test_loader:
                images, actions = images.to(device), actions.to(device)
                preds = model(images)
                val_loss += loss_fn(preds, actions).item() * images.size(0)
        val_loss /= len(test_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f"car_cnn_model_batch{batch_size}_epoch{epochs}_lr{str(learning_rate).replace('.', '')}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"---- Saved new best model: {model_path}")

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    #model_path = os.path.join(output_dir, f"car_cnn_model_batch{batch_size}_epoch{epochs}_lr{str(learning_rate).replace('.', '')}.pth")
    #torch.save(model.state_dict(), model_path)
    # Save loss plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve (batch={batch_size}, lr={learning_rate})')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(output_dir, f"loss_batch{batch_size}_epoch{epochs}_lr{str(learning_rate).replace('.', '')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot: {plot_path}")

# === Grid search setup ===
if __name__ == '__main__':
    data_file = "./car_caring_data_fixed_seed.pkl"  # path to your dataset
    output_dir = "models_2"
    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.00005, 0.0001, 0.0005, 0.0025, 0.005, 0.001]
    epochs_list = [500, 1000, 1500]
    processes = []

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for epochs in epochs_list:
                p = Process(target=train_model, args=(batch_size, lr, epochs, data_file, output_dir))
                p.start()
                processes.append(p)

    for p in processes:
        p.join()
