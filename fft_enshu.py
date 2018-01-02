#! /usr/bin/evn python3
# -*- using: utf-8 -*-

import numpy as np
import time
import functools
import matplotlib.pyplot as plt

def w(h, k):
    return np.exp(- 2j * np.pi * h / (2 ** k))

# K = 2 ** kのDFT
def dft(a, k):
    res = []
    for h in range(2**k):
        pk = 0
        for i in range(2**k):
            pk += a[i] * (w(h, k) ** i)
        res.append(pk)
    return np.array(res)

def fft(a, k):
    if k == 0:
        return a
    q = []
    s = []
    for i in range(0, 2**k, 2):
        q.append(a[i])
        s.append(a[i+1])

    q = fft(q, k-1)
    s = fft(s, k-1)

    for h in range(2 ** (k-1)):
        a[h] = q[h] + w(h, k) * s[h]
        a[2**(k-1)+h] = q[h] - w(h, k) * s[h]
    return a

dft_times = []
fft_times = []

for k in range(1, 11):
    data = np.random.rand(2**k) * (2 + 3j)
    start = time.time()
    rdft = dft(data, k)
    elapsed_time = time.time() - start
    dft_times.append(elapsed_time)
    start = time.time()
    rfft = fft(data, k)
    elapsed_time = time.time() - start
    fft_times.append(elapsed_time)

ks = np.array([2**k for k in range(1, 11)])

dft_line, = plt.plot(ks, np.array(dft_times), label="DFT")
fft_line, = plt.plot(ks, np.array(fft_times), label="FFT")
order_line = plt.plot(ks, ks*np.log(ks)*fft_times[0], linestyle='dashed', label="O(T log T)")

plt.legend()
# plt.xscale('log')
# plt.yscale('log')
plt.title('DFT and FFT')
plt.xlabel('T')
plt.ylabel('time(sec)')

plt.savefig('fft_enshu.png')