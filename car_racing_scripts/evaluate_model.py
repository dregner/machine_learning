import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
with open("./car_caring_data_fixed_seed.pkl", "rb") as f:
    data = pickle.load(f)

steer = [x["action"][0] for x in data]
gas   = [x["action"][1] for x in data]
brake = [x["action"][2] for x in data]
gas_brake = [gas[i] - brake[i] for i in range(len(gas))]
lap = [x["lap"] for x in data]
reward = [0] * (max(lap) + 1)

for i, l in enumerate(lap):
    reward[l] += data[i]["reward"]

# plt.figure(figsize=(12, 3))
# plt.subplot(1, 3, 1)
# plt.hist(steer, bins=50, color='blue'); plt.title("Steering")
# plt.subplot(1, 3, 2)
# plt.hist(gas, bins=50, color='green'); plt.title("Gas")
# plt.ylim(0, max(plt.gca().get_ylim()) * 0.5)  # Reduce the y-axis scale to 50% of its original
# plt.subplot(1, 3, 3)
# plt.hist(brake, bins=50, color='red'); plt.title("Brake")
# plt.tight_layout()
# plt.show()
import scipy.stats as stats

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
mean = np.mean(steer)
std = np.std(steer)

plt.hist(steer, bins=50, color='blue'); plt.title("Steering")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, p, 'k', linewidth=2, label=f"Mean: {mean:.2f}, Std: {std:.2f}")
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend()

plt.subplot(1, 3, 2)
mean = np.mean(gas_brake)
std = np.std(gas_brake)

plt.hist(gas_brake, bins=50, color='green', density=True, alpha=0.6); plt.title("Brake/Throttle")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, p, 'k', linewidth=2, label=f"Mean: {mean:.2f}, Std: {std:.2f}")
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend()
plt.subplot(1, 3, 3)
# plt.figure(figsize=(12, 3))
plt.bar(range(len(reward)), reward, color='blue'); plt.title("Score per lap")
plt.xlabel("Laps")
plt.tight_layout()

n = 500

# Plot three images from the dataset as subplots
plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)

    img = Image.open(data[n+i]["observation_path"]).convert("RGB")
    img = np.mean(np.array(img), axis=2, keepdims=True)  # Convert to grayscale
    plt.imshow(img, cmap='grey')  # Assuming each data entry has an "image" key with image data
    if i == 0:
        plt.title("Image N")
    else:
        plt.title(f"Image N+{i + 1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

