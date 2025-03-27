import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.classifyTools import Classifier


class Regression:
    def __init__(self, f, xs: np.array([]), ys: np.array([]), alpha: float, theta: np.array([])) -> None:
        """
        Liner/Polynomial Regression class
        :param f: true function
        :param xs: x samples
        :param alpha: learning rate
        :param theta: hypothesis parameters
        """
        self.f_true = f
        if len(xs.shape) < 2:
            self.size_xs = xs.shape[0]
            self.TH0 = []
            self.TH1 = []
            self.J_vals = []
        else:
            self.size_xs = xs.shape[1]

        self.ys = ys
        self.xs = np.array(xs)
        self.alpha = alpha
        self.theta = theta
        self.cost_value = []

    def h(self):
        """
        hypothesis function
        :return: transpose(theta) @ xs
        """
        if len(self.xs.shape) < 2:
            # X0 = ones
            x_components = [np.ones(self.size_xs)]

            # construct x components based on theta size
            for n in range(1, len(self.theta)):
                x_components.append(self.xs ** n)

            x = np.vstack(x_components)
        else:
            return 0

        # theta.T @ x
        h = self.theta[:, np.newaxis].T @ x
        return h[0]

    def J(self):
        """
        Cost function
        """
        return 1 / (2 * len(self.xs)) * np.sum((self.h() - self.ys) ** 2)

    def gradient(self, i):
        """
        Gradient function for each theta
        :param i: index of theta
        :return: value of new theta
        """
        if i == 0:
            return self.theta[i] - self.alpha * 1 / self.xs.shape[0] * np.sum(self.h() - self.ys)

        else:
            return self.theta[i] - self.alpha / self.xs.shape[0] * np.sum((self.h() - self.ys) * self.xs ** i)

    def print_modelo(self, k):
        """ plota no mesmo grafico : - o modelo / hipotese ( reta )
            -   a reta original ( true function )
            -   e os dados com ruido (xs , ys)
        """
        if len(self.theta) == 2:
            # ---- Left Plot: Model Fitting ----
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Two columns
            axes[0].scatter(self.xs, self.ys, label="Amostras")
            axes[0].plot(self.xs, self.f_true(self.xs), 'g', label="f_true(x)")
            axes[0].plot(self.xs, self.h(), 'r', label="h(x)")

            axes[0].set_xlabel('xs')
            axes[0].set_ylabel('ys')
            axes[0].legend()
            axes[0].set_title(f'Model fitting epoch {k}\n Learning rate: {self.alpha}')

            # Contour lines
            contours = axes[1].contour(self.TH0, self.TH1, self.J_vals, levels=50, colors=None)

            axes[1].clabel(contours, inline=True, fontsize=8)

            # Mark optimal theta
            a, b = np.unravel_index(np.argmin(self.J_vals), self.J_vals.shape)
            axes[1].scatter(self.TH0[a][b], self.TH1[a][b], color="red", marker="x", s=100, label="Optimal")
            axes[1].scatter(self.theta[0], self.theta[1], color="blue", marker="x", s=100, label="epoch: {}".format(k))

            # Labels
            axes[1].set_xlabel(r"$\theta_0$")
            axes[1].set_ylabel(r"$\theta_1$")
            axes[1].set_title("Cost Function Heatmap with Contours")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()  # Adjust spacing
            plt.show()

        else:
            plt.figure()
            plt.scatter(self.xs, self.ys)
            plt.plot(self.xs, self.f_true(self.xs), 'g')
            plt.plot(self.xs, self.h(), 'r')
            plt.xlabel('xs')
            plt.ylabel('ys')
            plt.legend(['amostras', 'f_true(x)', 'h(x)'])
            plt.title('Model fitting epoch {}\n learning rate: {}'.format(k, self.alpha))
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
        self.TH1 = TH1
        self.TH0 = TH0
        self.J_vals = J_vals
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
    """
    Linear function
    """
    return 2 + 0.8 * x


def f_true_2(x):
    """
    Polinomial function
    """
    return 2 + 0.8 * x + 2 * x ** 2 - 5 * x ** 3


def f_true_multi(x1, x2):
    """
    Polynomial function for multivariable input
    :param x1: variable 1
    :param x2: variable 2
    :return: function
    """
    return 7 - 0.3 * x1 + 1 * x1 * x2 - 3 * x2 ** 2 - 3 * x1 ** 2


def main():
    # Conjunto de dados {(x,y)}
    xs = np.linspace(-3, 3, 100)
    ys = np.array([f_true(x) + np.random.randn() * 1 for x in xs])
    theta = np.array([1, 1])

    # Hyper parameters
    alpha = 10e-4
    epoch = 5000
    n_plots = 5

    # ----------- POLINOMIAL/LINEAR REGRESSION ------------------
    # Construct regression class with theta = [th0, th1] to plot cost function surf
    if theta.shape[0] == 2:
        regression = Regression(f=f_true, xs=xs, ys=ys, alpha=alpha, theta=theta)
        regression.plot_cost_function_thetas()
    else:
        regression = Regression(f=f_true_2, xs=xs, ys=ys, alpha=alpha, theta=theta)

    # Begin epoch training
    for k in range(epoch):
        theta_epoch = np.zeros(theta.shape)
        regression.cost_value.append(regression.J())
        for i in range(theta.shape[0]):
            theta_epoch[i] = regression.gradient(i)
        regression.theta = theta_epoch
        if k % (epoch // n_plots) == 0:
            regression.print_modelo(k=k)
    regression.plot_cost_function()
    print('Polinomial theta values: {}'.format(regression.theta))


if __name__ == '__main__':
    print("Daniel Juchem Regner")
    main()
