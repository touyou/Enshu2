#! /usr/bin/env python3
# -*- using: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 厳密解
# y(x) = -1/2 * e^-x + 2/5 * e^-2x + 3/10 * sin(x) + 1/10 * cos(x)

# Felberg Butcher
A = [
[0, 0, 0, 0, 0, 0],
[1.0/4, 0, 0, 0, 0, 0],
[3.0/32, 9.0/32, 0, 0, 0, 0],
[1932.0/2197, -7200.0/2197, 7296.0/2197, 0, 0, 0],
[439.0/216, -8.0, 3680.0/513, -845.0/4104, 0, 0],
[-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40, 0]
]
c = [0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2]
b5 = [16.0/135, 0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55]
b4 = [25.0/216, 0, 1408.0/2565, 2197.0/4104, -1.0/5, 0]

def funy(x, y):
    return (-1 / 2.0) * np.exp(-x) + (2 / 5.0) * np.exp(-2 * x) + (3 / 10.0) * np.sin(x) + (1 / 10.0) * np.cos(x)

# dy/dx
def dy(x, y):
    return (1 / 2.0) * np.exp(-x) - (4 / 5.0) * np.exp(-2 * x) + (3 / 10.0) * np.cos(x) - (1 / 10.0) * np.sin(x)

# dg/dx
def dg(x, y):
    return np.cos(x) - 3 * dy(x, y) - 2 * y

e_tol = 1e-10
alpha = 0.5
x = 0
y = 0
g = 0
h = 0.01

fig, (axL, axM, axR) = plt.subplots(ncols=3, figsize=(20,4))

while x <= 2 * np.pi:
    while True:
        fy = [0, 0, 0, 0, 0, 0]
        fg = [0, 0, 0, 0, 0, 0]
        for k in range(6):
            fy[k] = dy(x + h * c[k], y + h * sum([A[k][j] * fy[j] for j in range(6)]))
            fg[k] = dg(x + h * c[k], y + h * sum([A[k][j] * fg[j] for j in range(6)]))
        ny1 = y + h * sum([b5[j] * fy[j] for j in range(6)])
        ny2 = y + h * sum([b4[j] * fy[j] for j in range(6)])
        ng1 = g + h * sum([b5[j] * fg[j] for j in range(6)])
        ng2 = g + h * sum([b4[j] * fg[j] for j in range(6)])

        T = ny1 - ny2
        if T >= e_tol:
            h = h / 2.0
        else:
            h = alpha * h * np.power(e_tol / np.abs(T), 1 / 5.0)
            y = ny1
            g = ng1
            x = x + h
            break
    axL.plot(x, y, c='red', marker='o')
    axM.plot(x, h, c='red', marker='o')
    axR.plot(x, np.abs(funy(x, y) - y), c='red', marker='o')

t = np.linspace(0, 2 * np.pi, 1000)
axL.plot(t, funy(t, t), linewidth=2)

axL.set_title('Runge-Kutta')
axM.set_title('Stride')
axR.set_title('Error')

plt.savefig('kadai4.png')
