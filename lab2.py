import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return np.where((0 <= x) & (x <= np.pi / 2), np.cos(x), 0)

def a_n(n):
    integrand = lambda x: f(x) * np.cos(n * x)
    return (2 / np.pi) * quad(integrand, 0, np.pi)[0]


def b_n(n):
    integrand = lambda x: f(x) * np.sin(n * x)
    return (2 / np.pi) * quad(integrand, 0, np.pi)[0]

def fourier_series_even(x, n_terms):
    result = 0.5 * a_n(0)
    for n in range(1, n_terms + 1):
        result += a_n(n) * np.cos(n * x)
    return result


def fourier_series_odd(x, n_terms):
    result = 0
    for n in range(1, n_terms + 1):
        result += b_n(n) * np.sin(n * x)
    return result


x = np.linspace(-np.pi, np.pi, 1000)

def plot_combined_fourier_series():
    n_values = [2, 5, 15]
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, n in enumerate(n_values):
        axs[0, i].plot(x, fourier_series_even(x, n), label=f'$f_e(x)$, n={n}')
        axs[0, i].axhline(0, color='black', linewidth=0.5)
        axs[0, i].axvline(0, color='black', linewidth=0.5)
        axs[0, i].grid(color='gray', linestyle='--', linewidth=0.5)
        axs[0, i].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axs[0, i].set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        axs[0, i].set_yticks([-1, 0, 1])
        axs[0, i].set_ylim(-1.5, 1.5)
        axs[0, i].set_xlabel('$x$')
        axs[0, i].set_ylabel('$f_e(x)$')
        axs[0, i].set_title(f'Четное продолжение $f_e(x)$, n={n}')
        axs[0, i].legend()

        axs[1, i].plot(x, fourier_series_odd(x, n), label=f'$f_o(x)$, n={n}')
        axs[1, i].axhline(0, color='black', linewidth=0.5)
        axs[1, i].axvline(0, color='black', linewidth=0.5)
        axs[1, i].grid(color='gray', linestyle='--', linewidth=0.5)
        axs[1, i].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axs[1, i].set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        axs[1, i].set_yticks([-1, 0, 1])
        axs[1, i].set_ylim(-1.5, 1.5)
        axs[1, i].set_xlabel('$x$')
        axs[1, i].set_ylabel('$f_o(x)$')
        axs[1, i].set_title(f'Нечетное продолжение $f_o(x)$, n={n}')
        axs[1, i].legend()

        axs[2, i].plot(x, f(x), label=f'$f(x)$, Original')
        axs[2, i].axhline(0, color='black', linewidth=0.5)
        axs[2, i].axvline(0, color='black', linewidth=0.5)
        axs[2, i].grid(color='gray', linestyle='--', linewidth=0.5)
        axs[2, i].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axs[2, i].set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        axs[2, i].set_yticks([-1, 0, 1])
        axs[2, i].set_ylim(-1.5, 1.5)
        axs[2, i].set_xlabel('$x$')
        axs[2, i].set_ylabel('$f(x)$')
        axs[2, i].set_title(f'Общий $f(x)$')
        axs[2, i].legend()

    plt.tight_layout()
    plt.show()



plot_combined_fourier_series()
