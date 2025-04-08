import numpy as np
import gymnasium as gym
import pygame
import pickle
from PIL import Image

def register_input(a, quit):
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

        if event.type == pygame.QUIT:
            quit = True
    return a, quit

if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False)

    pygame.init()  # Initialize pygame for rendering
    pygame.display.set_mode((600, 400))  # Set the display size
    pygame.display.set_caption("Car Racing")  # Set the window title

    observation, info = env.reset()  # Reset the environment to get the initial observation

    a = np.array([0.0, 0.0, 0.0])  # Initialize action array
    clock = pygame.time.Clock()  # Create a clock object to control the frame rate

    data = []
    quit = False
    step = 0
    while not quit:

        a, quit = register_input(a, quit) # Read action from keyboard

        observation, reward, terminated, truncated, info = env.step(a)  # Take action
        env.render()  # Render the environment

        done = terminated or truncated
        img = Image.fromarray(observation)
        img.save('frame_{}.png'.format(step))

            # Save frame data
        data.append({
            "observation": observation,
            "action": a.copy(),
            "reward": reward,
            "done": done
        })            

        step += 1
        clock.tick(60)  # Control the frame rate

with open('car_racing_data.pkl', 'wb') as f:
    pickle.dump(data, f)  # Save the data to a file

    env.close()
    pygame.quit()  # Quit pygame
