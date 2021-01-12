#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ANCHOR: mandelbrot
def mandelbrot(z, max_iterations):
    c = z
    for i in range(max_iterations):
        if abs(z) > 2:
            return i
        z = z*z + c
    return max_iterations
# ANCHOR_END: mandelbrot

width = 1000
height = 1000
max_iterations = 100
xmin = -2.5
xmax = 1.5
ymin = -2.0
ymax = 2.0

# Creates a list of equally distributed numbers
reals = np.linspace(xmin, xmax, width)
imaginaries = np.linspace(ymin, ymax, height) * 1j

# Creating a combination of all values in the two lists
zs = (reals+imaginaries[:, np.newaxis]).flatten().astype(np.complex64)

start_time = time.time()

res = np.vectorize(mandelbrot)(zs, max_iterations)

total_time = time.time() - start_time
print("Elapsed time:", total_time)

# Setting shape of array to help displaying it
res.shape = (width, height)

# Displaying the Mandelbrot set
fig, ax = plt.subplots()

ax.imshow(res, interpolation='bicubic', cmap=plt.get_cmap("terrain"))
plt.axis("off")
plt.tight_layout()

plt.show()
