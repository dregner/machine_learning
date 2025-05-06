import numpy as np
import matplotlib.pyplot as plt

# Data generation
mean0, std0 = -0.4, 0.5
mean1, std1 = 0.9, 0.3
m = 200

x1s = np.random.randn(m // 2) * std1 + mean1
x0s = np.random.randn(m // 2) * std0 + mean0
xs = np.hstack((x1s, x0s))
ys = np.hstack((np.ones(m // 2), np.zeros(m // 2)))

plt.plot(xs[:m//2], ys[:m//2], '.')
plt.plot(xs[m//2:], ys[m//2:], '.')
plt.title('Dados de treino')
plt.show()

# Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, theta):
    return sigmoid(theta[0] + theta[1] * x)

def cost(h_val, y):
    return -y * np.log(h_val) - (1 - y) * np.log(1 - h_val)

def J(theta, xs, ys):
    total_cost = 0
    for x, y in zip(xs, ys):
        total_cost += cost(h(x, theta), y)
    return total_cost / len(xs)

def gradient(i, theta, xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        if i == 0:
            grad += (h(x, theta) - y)
        else:
            grad += (h(x, theta) - y) * x
    return grad / len(xs)

def plot_fronteira(theta):
    # Fronteira é onde theta[0] + theta[1]*x = 0 => x = -theta[0]/theta[1]
    # Fronteira is where theta[0] + theta[1]*x = 0 => x = -theta[0]/theta[1]
    fronteira_x = -theta[0] / theta[1]
    # Define x range for the plot
    x_vals = np.linspace(xs.min(), xs.max(), 100)
    y_vals = theta[0] + theta[1] * x_vals  # Linear equation for the boundary
    plt.plot(x_vals, y_vals, color='red', label='Fronteira de decisão')

def print_modelo(theta, xs, ys):
    plt.subplot(1, 2, 1)
    plt.plot(xs[:m//2], ys[:m//2], '.')
    plt.plot(xs[m//2:], ys[m//2:], '.')
    plot_fronteira(theta)
    plt.title('Dados + Fronteira')

    plt.subplot(1, 2, 2)
    predictions = (h(xs, theta) >= 0.5).astype(int)
    plt.plot(xs[predictions==1], ys[predictions==1], 'go', label='Classe 1')
    plt.plot(xs[predictions==0], ys[predictions==0], 'ro', label='Classe 0')
    plt.legend()
    plt.title('Classificação')
    plt.show()

def accuracy(ys, predictions):
    num = sum(ys == predictions)
    return num / len(ys)

# Training setup
alpha = 5e3
epochs = 4000
theta = np.random.randn(2)

# Tracking metrics
costs = []
accuracies = []
fronteiras = []

for k in range(epochs):
    grads = np.array([gradient(0, theta, xs, ys), gradient(1, theta, xs, ys)])
    theta -= alpha * grads
    
    predictions = (h(xs, theta) >= 0.5).astype(int)
    acc = accuracy(ys, predictions)
    accuracies.append(acc)
    costs.append(J(theta, xs, ys))
    fronteiras.append(-theta[0]/theta[1])
    
    if k % 100 == 0:
        print(f"Epoch {k}, Acurácia: {acc:.4f}")

# Final visualization
print_modelo(theta, xs, ys)

# Plot Cost, Accuracy, Fronteira
fig, axs = plt.subplots(3, 1, figsize=(8, 6))

axs[0].plot(range(epochs), costs)
axs[0].set_ylabel('Custo')
# axs[0].set_title('Custo vs Épocas')

axs[1].plot(range(epochs), accuracies)
axs[1].set_ylabel('Acurácia')
# axs[1].set_title('Acurácia vs Épocas')

axs[2].plot(range(epochs), fronteiras)
axs[2].set_ylabel('Fronteira')
# axs[2].set_title('Fronteira vs Épocas')

plt.tight_layout()
plt.show()
