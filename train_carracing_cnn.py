import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# ===== Hyperparameters =====
BATCH_SIZE = 64
LEARNING_RATE = 0.00025
EPOCHS = 1000
TRAIN_SPLIT = 0.8
STACK_SIZE = 4
IMG_SIZE = (84, 84)
DATA_FILE = "./car_caring_data_fixed_seed.pkl"  # Path to the dataset file
# DATA_FILE = "data_francisco/car_racing_data_francisco.pkl"  # Path to the dataset file
MODEL_FILE = "car_cnn_model_gru_batch{}_epoch{}_lr{:.6f}.pth".format(
    BATCH_SIZE, EPOCHS, LEARNING_RATE)

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
            f = frame["observation_path"].split("\\")
            img = Image.open(f[0]+'/'+f[1]).convert("RGB")
            #img = Image.open(frame["observation_path"]).convert("RGB")
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
        
# ===== ResNet Policy (Optional) =====
class ResNetDrivingPolicy(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove final fc
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # steer, gas_brake
        )

    def forward(self, x):
        x = x / 255.0
        x = self.feature_extractor(x)
        return self.head(x)
# ===== Nature CNN =====

class NatureCNN(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # steer, gas_brake
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        return self.fc(x)

# ===== CNN + GRU Policy =====
class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=64*9*9, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        x = x.view(x.size(0), 1, -1)  # (batch, seq=1, features)
        x, _ = self.rnn(x)
        return self.fc(x[:, -1])
   
# ===== Load dataset =====
dataset = CarRacingFrameStackDataset(DATA_FILE, transform=transform, stack_size=STACK_SIZE)
train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# ===== Training setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_GRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.SmoothL1Loss()
best_val_loss = float("inf")  # <--- Add this before loop
train_score, val_score = [], []

# ===== Training loop =====
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    prev_preds = None

    for images, actions in train_loader:
        images, actions = images.to(device), actions.to(device)

        preds = model(images)
        loss = loss_fn(preds, actions)

        smoothness = ((preds[1:] - preds[:-1])**2).mean()
        # loss += 0.01 * smoothness

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

    if epoch % 50 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print("Sample pred:", preds[0].detach().cpu().numpy())
        print("Sample true:", actions[0].detach().cpu().numpy())

    # --- Save if 20% better ---
    # if val_loss < 0.7 * best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), f"car_cnn_model_epoch_{epoch}.pth")
    #     print(f"Model saved (val_loss improved to {val_loss:.6f}, 20% better than previous best)")
    # train_score.append(train_loss)
    # val_score.append(val_loss)
    val_score.append(val_loss)
    train_score.append(train_loss)
# ===== Save the model =====
torch.save(model.state_dict(), MODEL_FILE)
print(f"Done. Model saved as '{MODEL_FILE}'")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_score, label="Train Loss")
plt.plot(val_score, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(MODEL_FILE.replace(".pth", "_loss_plot.png"))  # Save the figure with the same name as the model
plt.show()  # Display the figure