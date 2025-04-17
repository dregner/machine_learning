import pickle
import matplotlib.pyplot as plt

with open("./car_caring_data_manuela.pkl", "rb") as f:
    data = pickle.load(f)

steer = [x["action"][0] for x in data]
gas   = [x["action"][1] for x in data]
brake = [x["action"][2] for x in data]

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.hist(steer, bins=50, color='blue'); plt.title("Steering")
plt.subplot(1, 3, 2)
plt.hist(gas, bins=50, color='green'); plt.title("Gas")
plt.subplot(1, 3, 3)
plt.hist(brake, bins=50, color='red'); plt.title("Brake")
plt.tight_layout()
plt.show()