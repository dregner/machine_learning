import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __init__(self, f, xs, alfa, theta):
        self.f_true = f
        self.xs = np.array(xs)
        self.ys = np.array([self.f_true(x) + np.random.randn() * 0.5 for x in xs])
        self.alfa = alfa
        self.theta = theta
        self.cost_value = []


    def h(self):
        h = self.theta[:,np.newaxis].T @ np.vstack((np.ones(self.xs.shape[0]), self.xs))
        return h[0]

    def J(self):
        return 1 / (2 * len(self.xs)) * np.sum((self.h() -self.ys) ** 2)


    def gradient(self, i):
        if i == 0:
            return self.theta[i] - self.alfa * 1 / self.xs.shape[0] * np.sum(self.h() - self.ys)

        else:
           return self.theta[i] - self.alfa / self.xs.shape[0] * np.sum((self.h() - self.ys)*self.xs)



    def print_modelo(self, k):
        """ plota no mesmo grafico : - o modelo / hipotese ( reta )
            -   a reta original ( true function )
            -   e os dados com ruido (xs , ys)
        """
        plt.figure()
        plt.scatter(self.xs, self.ys)
        plt.plot(self.xs, self.f_true(self.xs), 'g')
        plt.plot(self.xs, self.h(), 'r')
        plt.xlabel('xs')
        plt.ylabel('ys')
        plt.legend(['amostras', 'f_true(x)', 'h(x)'])
        plt.title('Model fitting epoch {}'.format(k))
        plt.show()

    def plot_cost_function(self):
        plt.figure()
        plt.plot(self.cost_value, '*')
        plt.plot(self.cost_value)
        plt.show()

    def plot_cost_function_thetas(self):
        theta0_vals = np.linspace(-10, 10, 100)
        theta1_vals = np.linspace(-10, 10, 100)
        TH0, TH1 = np.meshgrid(theta0_vals, theta1_vals)  # Create a grid of theta0, theta1

        # Store the original theta
        original_theta = self.theta.copy()

        # Compute cost function J for all values in grid
        J_vals = np.zeros_like(TH0)
        for i in range(TH0.shape[0]):
            for j in range(TH0.shape[1]):
                self.theta = np.array([TH0[i, j], TH1[i, j]])
                J_vals[i, j] = self.J()

        # Restore original theta
        self.theta = original_theta

        # Create 3D plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(TH0, TH1, J_vals, cmap="hsv", alpha=0.8)

        # Labels
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.set_zlabel(r"$J(\theta)$")
        ax.set_title("Cost Function Surface")

        plt.show()

def f_true(x):
    return 2 + 0.8 * x


def main():
    # Conjunto de dados {(x,y)}
    xs = np.linspace(-3, 3, 100)
    theta = np.array([1,1,1])

    # Hyper parameters
    alfa = 0.001
    epoch = 5000
    regression = Regression(f=f_true, xs=xs,alfa=alfa, theta=theta)
    cost_value = []
    regression.plot_cost_function_thetas()
    for k in range(epoch):
        theta_epoch = np.zeros(theta.shape)
        regression.cost_value.append(regression.J())
        for i in range(theta.shape[0]):
            theta_epoch[i] = regression.gradient(i)
        regression.theta = theta_epoch
        if k % 500 == 0:
            regression.print_modelo(k=k)
    regression.plot_cost_function()


    plt.figure()





if __name__ == '__main__':
    main()


