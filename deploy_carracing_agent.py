import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torchvision import transforms
from collections import deque
from PIL import Image
import pygame
def register_input(a, quit, automatic=False):
    key = pygame.key.get_pressed()  # Get the state of all keys
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            if event.key == pygame.K_ESCAPE:
                quit = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0
            if event.key == pygame.K_ESCAPE:
                quit = True
            if event.key == pygame.K_SPACE:
                automatic = not automatic
                print(f"Automatic mode: {automatic}")

        if event.type == pygame.QUIT:
            quit = True
    return a, quit, automatic

# ===== CNN model (same as training) =====
class CarRacingCNNPolicy(nn.Module):
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
        return self.fc(self.conv(x))

# ===== Load model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarRacingCNNPolicy().to(device)
model.load_state_dict(torch.load("car_cnn_model_epoch_377.pth", map_location=device))
model.eval()

# ===== Preprocessing =====
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

# ===== Environment setup =====
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.98, domain_randomize=False, continuous=True)
pygame.init()  # Initialize pygame for rendering
pygame.display.set_mode((600, 400))  # Set the display size
pygame.display.set_caption("Car Racing")  # Set the window title
clock = pygame.time.Clock()  # Create a clock object to control the frame rate

obs, _ = env.reset()

frame_stack = deque(maxlen=4)

frame_stack.clear()
for _ in range(4):
    obs, _,_,_,_ = env.step(np.zeros(3))
    gray = preprocess(obs)
    frame_stack.append(gray)
done = False
action1 = np.array([0.0, 0.0, 0.0])  # Initialize action array
automatic = False  # Flag for automatic mode
while not done:
    # Stack frames: [4, 96, 96]
    stacked_obs = torch.cat(list(frame_stack), dim=0).unsqueeze(0).to(device)  # [1, 4, 96, 96]
    action1, done, automatic = register_input(action1, done, automatic)  # Get user input
    # Predict action
    with torch.no_grad():
        output = model(stacked_obs).squeeze().cpu().numpy()
        steer, gas_brake = output
        steer = np.clip(steer, -1, 1)
        gas_brake = np.clip(gas_brake, -0.9, 1)

        # Decode gas/brake from single value
        if gas_brake >= 0:
            gas = gas_brake
            gas = max(gas, 0.1)
            brake = 0.0
        else:
            gas = 0.0
            brake = -gas_brake
        action = np.array([steer, gas, brake])


    print(f"Predicted action: {output}")    
    action_define = action if automatic else action1  # Use automatic action if enabled
    # Step environment
    obs, reward, terminated, truncated, _ = env.step(action_define)
    # done = terminated or truncated

    # Add new frame to stack
    gray = preprocess(obs)
    frame_stack.append(gray)
    clock.tick(60)  # Control frame rate
env.close()
print("Race finished.")
