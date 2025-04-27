import numpy as np
import gymnasium as gym
import pygame
import pickle
from PIL import Image
import os

def register_input(a, quit, steer_delta=0.2, gas_delta=0.2, brake_delta=0.1, decay=0.8):
    key = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit = True
        if event.type == pygame.QUIT:
            quit = True

    # --- Steering ---
    if key[pygame.K_LEFT]:
        a[0] -= steer_delta
    elif key[pygame.K_RIGHT]:
        a[0] += steer_delta
    else:
        a[0] *= decay  # gradual recenter

    a[0] = max(-1.0, min(a[0], 1.0))  # clamp steer

    # --- Gas ---
    if key[pygame.K_UP]:
        a[1] += gas_delta
    else:
        a[1] *= decay  # gradual release

    a[1] = max(0.0, min(a[1], 1.0))  # clamp gas

    # --- Brake ---
    if key[pygame.K_DOWN]:
        a[2] += brake_delta
    else:
        a[2] *= decay  # gradual release

    a[2] = max(0.0, min(a[2], 1.0))  # clamp brake

    return a, quit


if __name__ == "__main__":
    continuous = True
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.98, domain_randomize=False, continuous=continuous)
    env.reset()  # Set the seed for reproducibility
    IMG_DIR = 'data_fixed_seed'  # Directory to save images
    os.makedirs(IMG_DIR, exist_ok=True)  # Create directory for images if it doesn't exist
    NUM_LAPS = 5 # Number of laps to run
    lap_counter = 0
    step_counter = 0

    pygame.init()  # Initialize pygame for rendering
    pygame.display.set_mode((1200, 800))  # Set the display size
    pygame.display.set_caption("Car Racing")  # Set the window title

    observation, info = env.reset(seed=123)  # Reset the environment to get the initial observation
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Initialize action array

    clock = pygame.time.Clock()  # Create a clock object to control the frame rate

    data = []
    quit = False
    step = 0
    while not quit:

        a, quit = register_input(a, quit) # Read action from keyboard


        observation, reward, terminated, truncated, info = env.step(a)  # Take action
        env.render()  # Render the environment

        done = terminated or truncated
        # Save image + metadata
        img_path = os.path.join(IMG_DIR, f"lap{lap_counter:02d}_step{step_counter:04d}.png")
        Image.fromarray(observation).save(img_path)

        data.append({
            "lap": lap_counter,
            "step": step_counter,
            "observation_path": img_path,
            "action": a.copy(),
            "reward": reward,
            "done": done
        })

        step_counter += 1

        # Check if lap (episode) is done
        if done:
            lap_counter += 1
            print(f"Lap {lap_counter} complete.")
            step_counter = 0
            if lap_counter >= NUM_LAPS:
                quit = True
            else:
                obs, _ = env.reset(seed=42)
 
        clock.tick(60)  # Control the frame rate

env.close()
pygame.quit()  # Quit pygame

with open('car_caring_data_fixed_sd.pkl', 'wb') as f:
    pickle.dump(data, f)  # Save the data to a file
