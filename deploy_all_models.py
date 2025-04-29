import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Define CNN policy architectures
import torch.nn as nn
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
   
# Utility to select model based on filename

def load_model(path, device):
    # Choose architecture by model name
    if 'resnet' in os.path.basename(path):
        model = ResNetDrivingPolicy()
    elif 'gru' in os.path.basename(path):
        model = CNN_GRU()
    elif 'nature' in os.path.basename(path):
        model = NatureCNN()
    else:
        model = CarRacingCNNPolicy()
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def evaluate_model(model, env, episodes=5, device='cuda'):
    scores = []
    preprocess = lambda obs: torch.from_numpy(obs).float().to(torch.float32)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        # initialize frame stack
        stack = deque(maxlen=4)
        gray = obs.mean(axis=2, keepdims=True)  # simple grayscale
        frame = torch.from_numpy(gray.transpose(2,0,1)).to(torch.float32).to(device)
        for _ in range(4): stack.append(frame)

        while not done:
            x = torch.cat(list(stack), dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x).squeeze().cpu().numpy()
                steer, gas_brake = out
                steer = np.clip(steer, -1, 1)
                gas_brake = np.clip(gas_brake, -0.9, 1)
            # decode action
            if gas_brake >= 0:
                gas = gas_brake
                gas = max(gas, 0.1)
                brake = 0.0
            else:
                gas = 0.0
                brake = -gas_brake
            action = np.array([steer, gas, brake])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            gray = obs.mean(axis=2, keepdims=True)
            frame = torch.from_numpy(gray.transpose(2,0,1)).to(torch.float32).to(device)
            stack.append(frame)
        scores.append(ep_reward)
    return np.mean(scores), np.std(scores)

# Main evaluation loop

def main(models_dir, episodes=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CarRacing-v3', render_mode=None)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    results = {}

    for mf in sorted(model_files):
        path = os.path.join(models_dir, mf)
        print(f"Evaluating {mf}...")
        model = load_model(path, device)
        mean_score, std_score = evaluate_model(model, env, episodes)
        results[mf] = (mean_score, std_score)
        print(f" -> {mf}: {mean_score:.2f} ± {std_score:.2f}")

    env.close()

    # Plot results
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]

    plt.figure(figsize=(10,6))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Score')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate multiple CarRacing models')
    parser.add_argument('--models_dir', type=str, default='./models_2', help='Directory containing .pth model files')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes per model')
    args = parser.parse_args()
    main(args.models_dir, args.episodes)
