import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
from torchvision import transforms

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
        

 # ===== Preprocessing =====
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])  
# Utility to select model based on filename

def load_model(path, device):

    model = CarRacingCNNPolicy().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def evaluate_model(model, env, episodes=5, device='cuda'):
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        # Initialize frame stack
        frame_stack = deque(maxlen=4)
        frame_stack.clear()
        for _ in range(4):
            obs, _, _, _, _ = env.step(np.zeros(3))
            gray = preprocess(obs).to(device)
            frame_stack.append(gray)

        while not done:
            # Stack frames
            stacked_obs = torch.cat(list(frame_stack), dim=0).unsqueeze(0).to(device)
            # Predict action
            with torch.no_grad():
                output = model(stacked_obs).squeeze().cpu().numpy()
                steer, gas_brake = output
                steer = np.clip(steer, -1, 1)
                gas_brake = np.clip(gas_brake, -1, 1)

                # Decode gas/brake from single value
                if gas_brake >= 0:
                    gas = gas_brake
                    gas = max(gas, 0.1)
                    brake = 0.0
                else:
                    gas = 0.0
                    brake = -gas_brake
                action = np.array([steer, gas, brake])

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Add new frame to stack
            gray = preprocess(obs).to(device)
            frame_stack.append(gray)
        print(f"Episode {ep + 1}/{episodes} - Reward: {ep_reward:.2f}")
        scores.append(ep_reward)
    return np.mean(scores), np.std(scores)
# Main evaluation loop


def main(models_dir, episodes=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_env = gym.make('CarRacing-v3', render_mode=None, lap_complete_percent=0.9, domain_randomize=False)
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=1000)  # Adjust max_episode_steps as needed
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    results = {}

    for mf in sorted(model_files):
        path = os.path.join(models_dir, mf)
        print(f"Evaluating {mf}...")
        model = load_model(path, device)
        mean_score, std_score = evaluate_model(model, env, episodes, device)
        results[mf] = (mean_score, std_score)
        print(f" -> {mf}: {mean_score:.2f} ± {std_score:.2f}")

    env.close()

    # Process results for plotting
    grouped_results = {}
    for mf, (mean, std) in results.items():
        if 'car_cnn_model_' in mf:
            label = mf.split('car_cnn_model_')[-1].split('.')[0]
            if '_epoch' in mf:
                epoch = mf.split('_epoch')[-1].split('_')[0]
                if epoch not in grouped_results:
                    grouped_results[epoch] = []
                grouped_results[epoch].append((label, mean, std))
            else:
                print(f"Warning: Skipping file {mf} as it does not contain '_epoch'")

    for epoch, data in grouped_results.items():
        labels, means, stds = zip(*data)
        plt.figure(figsize=(10, 6))
        plt.bar(labels, means, yerr=stds, capsize=5)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Score')
        plt.title(f'Model Performance Comparison (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(f'model_comparison_epoch_{epoch}.png')
    with open('evaluation_results_noseed.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate multiple CarRacing models')
    parser.add_argument('--models_dir', type=str, default='./', help='Directory containing .pth model files')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes per model')
    args = parser.parse_args()
    main(args.models_dir, args.episodes)
