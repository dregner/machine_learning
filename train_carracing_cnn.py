import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ===== Hyperparameters =====
BATCH_SIZE = 128
LEARNING_RATE = 0.00025
EPOCHS = 400
TRAIN_SPLIT = 0.8
STACK_SIZE = 4
IMG_SIZE = (84, 84)
DATA_FILE = "./car_caring_data_manuela.pkl"  # Path to the dataset file
# DATA_FILE = "data_francisco/car_racing_data_francisco.pkl"  # Path to the dataset file
MODEL_FILE = "car_cnn_model.pth"

# ===== Transform: Grayscale + Resize + Tensor =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# ===== Dataset Class (Stack 4 Grayscale Frames) =====
class CarRacingFrameStackDataset(Dataset):
    def __init__(self, data_path, transform=None, stack_size=4):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.stack_size = stack_size

        # Organize by laps
        self.frames_by_lap = {}
        for item in self.data:
            lap = item["lap"]
            self.frames_by_lap.setdefault(lap, []).append(item)

        # Sort steps in each lap
        for lap in self.frames_by_lap:
            self.frames_by_lap[lap].sort(key=lambda x: x["step"])

        # Build frame sequences of length stack_size
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
            img = Image.open(frame["observation_path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            stacked.append(img)
        image_stack = torch.cat(stacked, dim=0)  # shape: [4, 84, 84]
        steer = frame_seq[-1]["action"][0]
        gas = frame_seq[-1]["action"][1]
        brake = frame_seq[-1]["action"][2]

        # Convert gas/brake into 1 value (gas-brake scale)
        gas_brake = gas - brake  # range [-1, 1]

        action = torch.tensor([steer, gas_brake], dtype=torch.float32)
        return image_stack, action

# ===== CNN Policy =====
class CarRacingCNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # [4,84,84] → [32,20,20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [32,20,20] → [64,9,9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),  # [64,9,9] → [64,4,4]
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # Output: steer(left/right), gas/brake
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        return self.fc(x)

# ===== Load dataset =====
dataset = CarRacingFrameStackDataset(DATA_FILE, transform=transform, stack_size=STACK_SIZE)
train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# ===== Training setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarRacingCNNPolicy().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# ===== Training loop =====
for epoch in range(1, EPOCHS + 1):
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

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, actions in test_loader:
            images, actions = images.to(device), actions.to(device)
            preds = model(images)
            loss = loss_fn(preds, actions)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(test_loader.dataset)

    if epoch % 100 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ===== Save the model =====
torch.save(model.state_dict(), MODEL_FILE)
print(f"Done. Model saved as '{MODEL_FILE}'")
