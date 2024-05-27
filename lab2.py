import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.where((x >= 0) & (x < np.pi / 2), np.cos(x), 0)


def a_k(k, x):
    if k == 0:
        return (1 / np.pi) * np.trapz(f(x), x)
    else:
        return (2 / np.pi) * np.trapz(f(x) * np.cos(k * x), x)


def b_k(k, x):
    if k == 0:
        return 0
    else:
        return (2 / np.pi) * np.trapz(f(x) * np.sin(k * x), x)


x = np.linspace(0, 2 * np.pi, 1000)


def partial_sum_cosines(n, x):
    a0 = a_k(0, x)
    sum_cos = a0 / 2 * np.ones_like(x)
    for k in range(1, n + 1):
        ak = a_k(k, x)
        sum_cos += ak * np.cos(k * x)
    return sum_cos


def partial_sum_sines(n, x):
    sum_sin = np.zeros_like(x)
    for k in range(1, n + 1):
        bk = b_k(k, x)
        sum_sin += bk * np.sin(k * x)
    return sum_sin


def partial_sum_total(n, x):
    a0 = a_k(0, x)
    sum_total = a0 / 2 * np.ones_like(x)
    for k in range(1, n + 1):
        ak = a_k(k, x)
        bk = b_k(k, x)
        sum_total += ak * np.cos(k * x) + bk * np.sin(k * x)
    return sum_total


def plot_partial_sums(n):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x, partial_sum_total(n, x), label=f'n = {n}')
    plt.plot(x, f(x), label='f(x)', linestyle='--')
    plt.title('Общая сумма')
    plt.xlabel('x')
    plt.ylabel('Sum')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x, partial_sum_cosines(n, x), label=f'n = {n}')
    plt.plot(x, f(x), label='f(x)', linestyle='--')
    plt.title('Сумма по cos')
    plt.xlabel('x')
    plt.ylabel('Sum')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x, partial_sum_sines(n, x), label=f'n = {n}')
    plt.plot(x, f(x), label='f(x)', linestyle='--')
    plt.title('Сумма по sin')
    plt.xlabel('x')
    plt.ylabel('Sum')
    plt.legend()

    plt.tight_layout()
    plt.show()


n = 10
plot_partial_sums(n)
